# 导入必要的模块
import asyncio, warnings, copy, time

class BaseNode:
    """基础节点类，提供节点执行的基本框架"""
    def __init__(self):
        self.params, self.successors = {}, {}  # 参数和后续节点字典
    def set_params(self, params):
        self.params = params  # 设置节点参数
    def next(self, node, action="default"):
        # 设置后续节点，如果已存在则发出警告
        if action in self.successors:
            warnings.warn(f"Overwriting successor for action '{action}'")
        self.successors[action] = node
        return node
    def prep(self, shared): pass  # 准备阶段，子类可重写
    def exec(self, prep_res): pass  # 执行阶段，子类可重写
    def post(self, shared, prep_res, exec_res): pass  # 后处理阶段，子类可重写
    def _exec(self, prep_res):
        return self.exec(prep_res)  # 内部执行方法
    def _run(self, shared):
        p = self.prep(shared)  # 准备
        e = self._exec(p)      # 执行
        return self.post(shared, p, e)  # 后处理
    def run(self, shared):
        # 运行节点，如果有后续节点则发出警告
        if self.successors:
            warnings.warn("Node won't run successors. Use Flow.")
        return self._run(shared)
    def __rshift__(self, other):
        return self.next(other)  # 重载右移操作符用于连接节点
    def __sub__(self, action):
        # 重载减法操作符用于条件转换
        if isinstance(action, str):
            return _ConditionalTransition(self, action)
        raise TypeError("Action must be a string")

class _ConditionalTransition:
    """条件转换类，用于处理条件分支"""
    def __init__(self, src, action):
        self.src, self.action = src, action
    def __rshift__(self, tgt):
        return self.src.next(tgt, self.action)

class Node(BaseNode):
    """标准节点类，支持重试机制"""
    def __init__(self, max_retries=1, wait=0):
        super().__init__()
        self.max_retries, self.wait = max_retries, wait
    def exec_fallback(self, prep_res, exc):
        raise exc  # 执行失败时的回退处理
    def _exec(self, prep_res):
        # 带重试机制的执行
        for self.cur_retry in range(self.max_retries):
            try:
                return self.exec(prep_res)
            except Exception as e:
                if self.cur_retry == self.max_retries - 1:
                    return self.exec_fallback(prep_res, e)
                if self.wait > 0:
                    time.sleep(self.wait)

class BatchNode(Node):
    """批处理节点类，支持批量处理"""
    def _exec(self, items):
        return [super(BatchNode, self)._exec(i) for i in (items or [])]

class Flow(BaseNode):
    """流程控制类，管理节点间的执行顺序"""
    def __init__(self, start=None):
        super().__init__()
        self.start_node = start
    def start(self, start):
        self.start_node = start
        return start
    def get_next_node(self, curr, action):
        # 根据动作获取下一个节点
        nxt = curr.successors.get(action or "default")
        if not nxt and curr.successors:
            warnings.warn(f"Flow ends: '{action}' not found in {list(curr.successors)}")
        return nxt
    def _orch(self, shared, params=None):
        # 流程编排：按顺序执行节点链
        curr, p, last_action = copy.copy(self.start_node), (params or {**self.params}), None
        while curr:
            curr.set_params(p)
            last_action = curr._run(shared)
            curr = copy.copy(self.get_next_node(curr, last_action))
        return last_action
    def _run(self, shared):
        p = self.prep(shared)
        o = self._orch(shared)
        return self.post(shared, p, o)
    def post(self, shared, prep_res, exec_res):
        return exec_res

class BatchFlow(Flow):
    """批处理流程类，支持批量参数处理"""
    def _run(self, shared):
        pr = self.prep(shared) or []
        for bp in pr:
            self._orch(shared, {**self.params, **bp})
        return self.post(shared, pr, None)

class AsyncNode(Node):
    """异步节点类，支持异步操作"""
    async def prep_async(self, shared): pass  # 异步准备
    async def exec_async(self, prep_res): pass  # 异步执行
    async def exec_fallback_async(self, prep_res, exc):
        raise exc  # 异步执行失败回退
    async def post_async(self, shared, prep_res, exec_res): pass  # 异步后处理
    async def _exec(self, prep_res):
        # 异步重试执行
        for i in range(self.max_retries):
            try:
                return await self.exec_async(prep_res)
            except Exception as e:
                if i == self.max_retries - 1:
                    return await self.exec_fallback_async(prep_res, e)
                if self.wait > 0:
                    await asyncio.sleep(self.wait)
    async def run_async(self, shared):
        # 异步运行，有后续节点时发出警告
        if self.successors:
            warnings.warn("Node won't run successors. Use AsyncFlow.")
        return await self._run_async(shared)
    async def _run_async(self, shared):
        p = await self.prep_async(shared)
        e = await self._exec(p)
        return await self.post_async(shared, p, e)
    def _run(self, shared):
        raise RuntimeError("Use run_async.")  # 强制使用异步方法

class AsyncBatchNode(AsyncNode, BatchNode):
    """异步批处理节点类"""
    async def _exec(self, items):
        return [await super(AsyncBatchNode, self)._exec(i) for i in items]

class AsyncParallelBatchNode(AsyncNode, BatchNode):
    """异步并行批处理节点类"""
    async def _exec(self, items):
        return await asyncio.gather(*(super(AsyncParallelBatchNode, self)._exec(i) for i in items))

class AsyncFlow(Flow, AsyncNode):
    """异步流程控制类"""
    async def _orch_async(self, shared, params=None):
        # 异步流程编排
        curr, p, last_action = copy.copy(self.start_node), (params or {**self.params}), None
        while curr:
            curr.set_params(p)
            # 根据节点类型选择同步或异步执行
            last_action = await curr._run_async(shared) if isinstance(curr, AsyncNode) else curr._run(shared)
            curr = copy.copy(self.get_next_node(curr, last_action))
        return last_action
    async def _run_async(self, shared):
        p = await self.prep_async(shared)
        o = await self._orch_async(shared)
        return await self.post_async(shared, p, o)
    async def post_async(self, shared, prep_res, exec_res):
        return exec_res

class AsyncBatchFlow(AsyncFlow, BatchFlow):
    """异步批处理流程类"""
    async def _run_async(self, shared):
        pr = await self.prep_async(shared) or []
        for bp in pr:
            await self._orch_async(shared, {**self.params, **bp})
        return await self.post_async(shared, pr, None)

class AsyncParallelBatchFlow(AsyncFlow, BatchFlow):
    """异步并行批处理流程类"""
    async def _run_async(self, shared):
        pr = await self.prep_async(shared) or []
        # 并行执行所有批次
        await asyncio.gather(*(self._orch_async(shared, {**self.params, **bp}) for bp in pr))
        return await self.post_async(shared, pr, None)