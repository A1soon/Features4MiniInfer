from datetime import time
from functools import partial
import heapq
from miniinfer.utils import get_logger

import torch
from typing import List, Tuple, Any, Optional
import time
import hashlib

from .interface import IPrefixCache, ITokenAllocator

logger = get_logger(__name__)

# 注意在本文件中：
#  key   = token ids
#  value = 对应 token 的 KV physical slot ids


def _key_match_page_size1(key0: List[int], key1: List[int]) -> int:
  """逐 token 比较两个 key，返回公共前缀长度。"""
  i = 0
  for k0, k1 in zip(key0, key1):
    if k0 != k1:
      break
    i += 1
  return i


def _key_match_paged(key0: List[int], key1: List[int], page_size: int) -> int:
  """
  按 page 比较两个 key，返回 page 对齐的公共前缀长度。

  page_size > 1 时，prefix cache 只保存完整 page。
  这样 allocator 回收/复用时不会出现“半页被两个请求共享”的复杂情况。
  """
  min_len = min(len(key0), len(key1))
  i = 0
  while i < min_len:
    if key0[i : i + page_size] != key1[i : i + page_size]:
      break
    i += page_size

  return i


def get_child_key(key: List[int], page_size: int = 1):
  """
  取 radix tree 的分支 key。

  page_size=1 时，树的每条边按单 token 区分；
  page_size>1 时，树的每条边按一个 page 的 token tuple 区分。


  在 Radix Tree 中，每个节点可能有多个子节点，我们需要一个键来区分不同的分支。这个函数根据 page_size 决定用什么作为分支键：

page_size == 1：分支键是单个 token ID（整数）。
page_size > 1：分支键是一个页的 token ID 元组（因为列表不能作为字典的键，必须转换为元组）。
这个键用于在树节点的子字典中查找或插入。
  """
  if page_size == 1:
    plain_key = key[0]
  else:
    plain_key = tuple(key[:page_size])
  return plain_key


class TreeNode:
  """
  Radix tree 节点。

  一个节点保存一段连续 token key，以及这段 token 对应的一段 KV 物理位置 value。
  多个节点从 root 到 leaf 拼起来，就是一个完整缓存前缀。
  """

  def __init__(self):
    '''
    字典，键为子节点的分支 key（由 get_child_key 生成），值为对应的 TreeNode 子节点。
当 page_size=1 时，键是单个 token ID（int）；当 page_size>1 时，键是一个 page 的 token 元组（tuple）。
通过这个字典可以快速定位子节点，实现 O(1) 的查找。
    '''
    self.children = {}  # {token_id 或 token tuple -> TreeNode}
    self.parent = None
    '''
    列表，存储该节点自己持有的 token IDs，不包括祖先节点的 token。
在 Radix Tree 中，路径是分段存储的，所以每个节点只记录自己那一段。例如 root 的子节点 A 的 key 可能为 [1,2,3]，而其子节点 B 的 key 为 [4,5]，那么从 root 到 B 的完整 key 就是 [1,2,3,4,5]
    '''
    self.key = []  # 这个节点自己持有的 token ids，不包括祖先节点。
    '''
    与 self.key 对齐的 KV cache 物理位置列表（即之前 alloc_extend / alloc_decode 返回的 out_indices）。
如果该节点当前没有被任何请求使用（即已被驱逐或尚未分配），value 为 None。
当节点被某个请求复用或新插入时，会填充 value；当引用计数归零且被驱逐时，会清空 value 并释放对应的物理页。
    '''
    self.value = None  # 与 self.key 对齐的一段 KV indices。
    self.lock_ref = 0  # 有多少正在运行的请求使用这个节点；>0 时不能驱逐。
    self.last_access_time = 0  # LRU 驱逐用的访问时间。
    self.hash_value = None  # List of SHA256 hex strings，当前实现中没有用到。

  '''
  @property 将 evicted 方法转换为只读属性，调用时无需括号：node.evicted。
  返回值是 self.value is None，即该节点是否已被驱逐。
  逻辑：如果 value 为 None，说明该节点没有对应的 KV 缓存，已经被驱逐或从未分配；否则，它仍持有 KV 缓存。
  '''
  @property
  def evicted(self):
    return self.value is None

  def __lt__(self, other: "TreeNode"):
    return self.last_access_time < other.last_access_time


class RadixCache(IPrefixCache):
  """
  Prefix cache 的 radix tree 实现。

  它缓存的是：
      token prefix -> KV cache 物理位置

  例子：
  - 请求 A: [10, 11, 12, 13]
  - 请求 B: [10, 11, 99]
  两个请求共享前缀 [10, 11]。RadixCache 会把共享部分只存一次。

  读这个类时抓住三件事：
  1. match_prefix(): 新请求进来时查“已有 KV 可以复用多少”。
  2. insert(): 请求完成/释放时，把它的 KV 写入 prefix cache。
  3. evict(): 显存不够时，按 LRU 驱逐未被 lock 的叶子节点。
  """

  def __init__(self, token_allocator: ITokenAllocator, page_size: int = 1):
    '''
    self.root 是 Radix Tree 的根节点，它不代表任何 token，只是一个虚拟起点。
将根节点的 lock_ref 设为 1，目的是防止根节点被驱逐（因为根节点没有 value，也没有实际意义，但作为树的入口必须常驻）。
self.root_node = self.root 保留了一个别名，可能是为了兼容旧代码或外部引用。

protected_size_：记录当前被锁定（lock_ref > 0）的节点所占用的 token 总数（或物理页数）。这些节点不能被驱逐。
evictable_size_：记录当前可驱逐（lock_ref == 0）的节点所占用的 token 总数。LRU 驱逐时从这个池子里选择。
time_counter：一个单调递增的计数器，用于给节点的 last_access_time 赋值。每次访问节点时，将当前 time_counter 赋给节点，然后 time_counter += 1，从而实现 LRU 的“最近访问”排序。
    '''
    self.root = TreeNode()
    self.page_size = page_size
    self.root.lock_ref = 1  # 根节点永不驱逐
    # keep legacy references used elsewhere
    self.root_node = self.root
    self.protected_size_ = 0
    self.token_allocator = token_allocator
    self.time_counter = 0
    self.evictable_size_ = 0

    '''
    (1) page_size == 1 时

    self.key_match_fn 直接指向 _key_match_page_size1，该函数逐 token 比较，返回 token 级公共前缀长度。
    self.get_child_key_fn 直接指向 get_child_key，但注意 get_child_key 有一个默认参数 page_size=1，因此调用时可以不传 page_size，行为正确。
    (2) page_size > 1 时

  self.key_match_fn = partial(_key_match_paged, page_size=self.page_size)
self.get_child_key_fn = partial(get_child_key, page_size=self.page_size)
这里使用了 Python 标准库 functools.partial。partial 的作用是预先绑定函数的部分参数，生成一个新的可调用对象。例如：

python
from functools import partial
def f(a, b, c):
    ...
g = partial(f, b=2, c=3)
g(1)  # 等价于 f(1, 2, 3)
在这里，_key_match_paged 需要三个参数 (key0, key1, page_size)，通过 partial 固定 page_size=self.page_size，得到一个新函数，以后只需传入两个 key 即可调用。get_child_key 同理，它原有两个参数 (key, page_size)，固定后变成只需一个参数。

这样设计的好处是：在后续代码中，无论是 page_size=1 还是 >1，都可以用统一的接口调用 self.key_match_fn(key0, key1) 和 self.get_child_key_fn(key)，无需写条件分支。
    '''
    if self.page_size == 1:
      # 单 token radix tree，方便理解，但 page 化收益较小。
      self.key_match_fn = _key_match_page_size1
      self.get_child_key_fn = get_child_key
    else:
      # page 级 radix tree：只按完整 page 匹配/插入。
      logger.debug("Using paged radix tree with page size %d", self.page_size)
      self.key_match_fn = partial(_key_match_paged, page_size=self.page_size)
      self.get_child_key_fn = partial(get_child_key, page_size=self.page_size)
    if self.token_allocator is not None:
      self.device = self.token_allocator.device
    else:
      self.device = torch.device("cpu")

  def match_prefix(self, key: List[int]) -> Tuple[torch.Tensor, "TreeNode"]:
    """
    查找 key 在 prefix cache 中命中的最长前缀。

    返回：
    - value: 命中前缀对应的一维 KV indices tensor。
    - node: 命中的最后一个 radix tree 节点。
    返回node的另一个作用是方便后续插入新的token，即可以从这个node开始继续生长

    Scheduler 会把 value 写到 req.prefix_indices，把 node 写到 req.last_node。
    随后 KVCacheManager 会 lock 这个 node，防止它在请求执行期间被 evict。
    """
    if len(key) == 0:
      return torch.empty((0,), dtype=torch.int64), self.root
    if self.page_size > 1:
      # page 模式下只能复用完整 page，所以向下对齐。
      page_aligned_len = self._align_len(len(key))
      key = key[:page_aligned_len]
    node = self.root
    # time.monotonic() 返回一个单调递增的时钟值（秒），用于更新节点的最近访问时间（LRU 依据）
    access_time = time.monotonic()
    node.last_access_time = access_time

    child_key = self.get_child_key_fn(key)

    value = []
    # children是一个字典，keys()是这个字典自带的方法，其作用是获取该字典中所有的key
    # 即键值
    # 注意这里的node.children.keys()并不是访问children的成员变量key
    # 因为children是字典，这里利用字典自带的keys()方法访问字典的所有key
    # 同理若用children.vlaues()变量则访问的是字典中的valude这里的valude就是子节点
    while len(key) > 0 and child_key in node.children.keys():
      child = node.children[child_key]
      child.last_access_time = access_time

      # prefix_len 是当前 child.key 和剩余 key 的公共前缀长度。
      prefix_len = self.key_match_fn(child.key, key)
      if prefix_len < len(child.key):
        # 当前 child 节点只被部分命中，需要拆节点：
        # child: [公共前缀 + 分歧后缀]
        # 拆成 new_node: [公共前缀] -> child: [分歧后缀]
        new_node = self._split_node(child.key, child, prefix_len)
        value.append(new_node.value)
        node = new_node
        break
      else:
        # 当前 child 完全命中，继续往下找更长前缀。
        value.append(child.value)
        node = child
        key = key[prefix_len:]

        if len(key):
          child_key = self.get_child_key_fn(key)
    if value:
      value = torch.cat(value)
    else:
      value = torch.empty((0,), dtype=torch.int64, device=self.device)
    return value, node

  def insert(self, key: List[int], value=None):
    """
    把一段 token 序列和对应 KV indices 插入 prefix cache。

    key 是 token ids，value 是与 key 对齐的 KV 物理位置。
    如果 value=None，则默认把 key 自己当作 value，主要用于文件底部的简单测试。

    返回值是 insert 前已经存在的前缀长度。KVCacheManager 用它判断哪些新分配的
    KV page 是重复的，可以释放回 allocator。
    """
    # In paged mode we can only safely own full pages in radix cache.
    # Keep insert aligned with match_prefix semantics to avoid dangling refs.
    if self.page_size > 1:
      # page 模式只插入完整 page。末尾不满 page 的 token 仍归请求所有，
      # 请求释放时需要直接归还给 allocator。
      aligned_len = self._align_len(len(key))
      if aligned_len == 0:
        return 0
      if aligned_len != len(key):
        key = key[:aligned_len]
        if value is not None:
          value = value[:aligned_len]
    if value is None:
      # 优化：使用 torch.as_tensor 避免 CUDA 同步
      value_cpu = torch.as_tensor(key, dtype=torch.int64)
      value = value_cpu.to(self.device, non_blocking=True)
    return self._insert_helper(self.root, key, value)

  def evict(self, num_tokens: int):
    """
    从 prefix cache 中驱逐至少 num_tokens 个可驱逐 token。

    只会驱逐 leaf 且 lock_ref == 0 的节点。
    使用 last_access_time 组成最小堆，近似 LRU：越久没被访问越先驱逐。
    """
    leaves = self._collect_leaves()
    # 创建(last_access_time和node的二维元组列表)
    eviction_heap = [(node.last_access_time, node) for node in leaves]
    # 创建一个最小堆，堆顶是一个last_access_time最小的node
    heapq.heapify(eviction_heap)

    num_evicted = 0
    while num_evicted < num_tokens and len(eviction_heap):
      _priority, x = heapq.heappop(eviction_heap)

      # x.value 是这个节点持有的 KV indices。
      # 释放给 token_allocator 后，这些 page 才能重新分配给新请求。
      before_available = self.token_allocator.available_size()
      self.token_allocator.free(x.value)
      after_available = self.token_allocator.available_size()
      # Count net newly available token slots instead of inferred pages.
      # This avoids overcounting when stale/duplicate page refs exist.
      num_evicted += max(0, after_available - before_available)
      self._delete_leaf(x)

      if len(x.parent.children) == 0 and x.parent.lock_ref == 0:
        # 删除 leaf 后，父节点也可能变成可驱逐 leaf。
        new_priority = x.parent.last_access_time
        heapq.heappush(eviction_heap, (new_priority, x.parent))
    # self.pretty_print()
    logger.debug(f"Evicted {num_evicted} tokens, requested {num_tokens}")

  def inc_lock_ref(self, node: TreeNode):
    """
    给 node 以及它的祖先增加引用计数。

    当一个请求复用了 prefix cache 中的 KV，这些节点就不能被驱逐。
    evictable_size_ 会减少，protected_size_ 会增加。
    """
    delta = 0
    while node != self.root_node:
      if node.lock_ref == 0:
        self.evictable_size_ -= len(node.key)
        self.protected_size_ += len(node.key)
        delta -= len(node.key)
      node.lock_ref += 1
      node = node.parent
    return delta

  def dec_lock_ref(self, node: TreeNode):
    """
    请求结束/释放后，减少 node 以及祖先的引用计数。

    当某个节点 lock_ref 从 1 变成 0，它重新变成可驱逐缓存。
    """
    delta = 0
    while node != self.root_node:
      if node.lock_ref == 1:
        self.evictable_size_ += len(node.key)
        self.protected_size_ -= len(node.key)
        delta += len(node.key)
      node.lock_ref -= 1
      if node.parent is None:
        assert node is self.root_node, f"This request holds the node from another tree"
      node = node.parent
    return delta

  def evictable_size(self):
    """返回当前可被驱逐的 prefix cache token 数，按 page_size 向下对齐。"""
    aligned = self.evictable_size_ // self.page_size * self.page_size
    if self.token_allocator is None:
      return aligned
    # Evictable cache must be bounded by currently used physical capacity.
    used = max(0, self.token_allocator.size - self.token_allocator.available_size())
    return min(aligned, used)

  def protected_size(self):
    # protected size 指正在被请求使用、不能驱逐的缓存大小。
    return self.protected_size_ // self.page_size * self.page_size

  def total_size(self):
    """遍历整棵树，统计 prefix cache 中保存的 token 数。"""
    total_size = 0
    stack = [self.root]
    while stack:
      current_node = stack.pop()
      total_size += len(current_node.value) if current_node.value is not None else 0
      for child in current_node.children.values():
        if child.evicted:
          continue
        stack.append(child)
    return total_size

  def pretty_print(self):
    self._print_helper(self.root, 0)
    logger.debug(f"#tokens: {self.total_size()}")

  def _align_len(self, length: int) -> int:
    """向下取整到 page_size 的倍数"""
    return (length // self.page_size) * self.page_size

  def _split_node(self, key: List[int], child: TreeNode, split_len: int):
    """
    把一个节点按 split_len 拆开。

    拆分前：
      parent -> child(key=[公共前缀 + 分歧后缀])

    拆分后：
      parent -> new_node(key=[公共前缀]) -> child(key=[分歧后缀])

    这个操作是 radix tree 能压缩公共前缀的关键。
    """
    new_node = TreeNode()
    new_node.children = {self.get_child_key_fn(key[split_len:]): child}
    new_node.parent = child.parent
    new_node.lock_ref = child.lock_ref
    new_node.key = child.key[:split_len]
    new_node.value = child.value[:split_len]
    child.parent = new_node
    child.key = child.key[split_len:]
    child.value = child.value[split_len:]
    new_node.parent.children[self.get_child_key_fn(key)] = new_node

    return new_node

  def _insert_helper(self, node: TreeNode, key: List[int], value):
    """
    从 node 开始插入 key/value。

    如果已有路径和 key 部分重合，就沿着已有路径走；
    遇到部分匹配时拆节点；
    走不下去时创建新节点。
    """
    access_time = time.monotonic()
    node.last_access_time = access_time
    if len(key) == 0:
      return 0
    child_key = self.get_child_key_fn(key)
    logger.debug("Inserting key with %d remaining tokens", len(key))
    total_prefix_len = 0

    # 注意：这里的node.children.keys()访问的并不是其成员变量key（存储的是token的物理槽位）
    # 这里是通过字典的keys()方法访问的字典中所有的value这里指的是所有子节点的node
    while len(key) > 0 and child_key in node.children.keys():
      node = node.children[child_key]
      node.last_access_time = access_time
      prefix_len = self.key_match_fn(node.key, key)
      total_prefix_len += prefix_len
      key = key[prefix_len:]
      value = value[prefix_len:]

      if prefix_len < len(node.key):
        # 部分匹配，需要把已有节点拆成“共享前缀 + 原后缀”。
        new_node = self._split_node(node.key, node, prefix_len)
        node = new_node

      if len(key):
        child_key = self.get_child_key_fn(key)
        logger.debug("Descending with %d remaining tokens", len(key))

    if len(key):
      # 剩余 key 在树上不存在，创建一个新 leaf。
      new_node = TreeNode()
      new_node.parent = node
      new_node.key = key
      new_node.value = value
      node.children[child_key] = new_node
      self.evictable_size_ += len(key)
      logger.debug("Created new node with %d tokens", len(new_node.key))
    # self.pretty_print()
    return total_prefix_len

  def _collect_leaves(self):
    """收集所有可驱逐 leaf 节点。"""
    ret_list = []
    stack = list(self.root_node.children.values())

    while stack:
      cur_node = stack.pop()
      if len(cur_node.children) == 0:
        if cur_node.lock_ref == 0:
          ret_list.append(cur_node)
      else:
        stack.extend(cur_node.children.values())

    return ret_list

  def _print_helper(self, node: TreeNode, indent: int):
    """Prints the radix tree in a human-readable format."""
    stack = [(node, indent)]
    while stack:
      current_node, current_indent = stack.pop()
      logger.debug(
        f"{' ' * current_indent} {len(current_node.key)} {current_node.key[:10]} r={current_node.lock_ref}"
      )
      for key, child in current_node.children.items():
        stack.append((child, current_indent + 2))

        assert key == self.get_child_key_fn(
          child.key
        ), f"{key=}, {self.get_child_key_fn(child.key)=}"

  def _delete_leaf(self, node):
    """从父节点 children 中删除一个 leaf，并更新 evictable_size_。"""
    key = self.get_child_key_fn(node.key)
    v = node.parent.children.pop(key, None)
    assert v == node, f"parent does not have child key, {key}"

    self.evictable_size_ -= len(node.key)


if __name__ == "__main__":
  tree = RadixCache(token_allocator=None, page_size=2)

  # Example token id sequences (as lists of ints)
  tree.insert([1, 2, 3])
  tree.insert([1, 2, 3])
  tree.insert([1, 2, 4, 5])
  tree.insert([1, 2, 4, 5, 6, 7])
  tree.insert([8, 9, 10, 11, 12])
  tree.pretty_print()

  logger.info(tree.match_prefix([1, 2, 3, 13, 14]))
  logger.info(tree.match_prefix([1, 2, 3]))
  logger.info(tree.match_prefix([8, 9, 10, 11, 12, 13]))

  tree2 = RadixCache(token_allocator=None, page_size=1)
  tree2.insert([1, 2, 3])
  tree2.insert([1, 2, 4, 5])
  tree2.insert([1, 2, 4, 5, 6, 7])
  tree2.insert([8, 9, 10, 11, 12])
  # tree2.pretty_print()
  logger.info(tree2.match_prefix([1, 2, 3, 13, 14]))
  logger.info(tree2.match_prefix([1, 2, 3]))
  logger.info(tree2.match_prefix([8, 9, 10, 11, 12, 13]))
