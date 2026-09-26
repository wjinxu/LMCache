# 薄壳，vLLM接口
# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import TYPE_CHECKING, Any, Optional

# Third Party
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput
import torch

# First Party
from lmcache.integration.vllm.vllm_v1_adapter import LMCacheConnectorV1Impl

if TYPE_CHECKING:
    # Third Party
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.request import Request

logger = init_logger(__name__)

# LMcache的连接器（继承自 vLLM的基类 KVConnectorBase_V1 规定任何想接入 vLLM 做 KV cache 卸载的系统都应该做一个标准的插头）
class LMCacheConnectorV1Dynamic(KVConnectorBase_V1):
    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: Optional[Any] = None,
    ):
    # 简单的薄壳，具体的实现在_lmcache_engine中
        if kv_cache_config is not None:
            super().__init__(
                vllm_config=vllm_config,
                role=role,
                kv_cache_config=kv_cache_config,
            )
        else:
            super().__init__(vllm_config=vllm_config, role=role)
        # 创建一个真实的 LMCacheConnectorV1Impl 用来实现
        self._lmcache_engine = LMCacheConnectorV1Impl(vllm_config, role, self)
    # 方法分为两组：
    # Worker-side侧 ： vLLM真正跑模型的时候调用，用处 ： 用来异步搬运 KV 进/出
    # Scheduler-side ： vLLM 决定怎么调度的时候，用处 ： 问 LMCache:这请求能复用多少缓存?

    # ==============================
    # Worker-side methods
    # ==============================
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        """
        Initialize with the KV caches. Useful for pre-registering the
        KV Caches in the KVConnector (e.g. for NIXL).

        Args: kv_caches:
            dictionary of layer names, kv cache
        """

        """
        [笔记] vLLM 启动时把每层 GPU KV 缓冲区的"地图"(层名->张量)交给 LMCache 备案
            目的: 后续搬运数据(尤其 NIXL 零拷贝/RDMA)需提前知道显存地址并注册


        用 KV caches 进行初始化。用于在 KVConnector 中预先注册
        KV Caches(例如给 NIXL 用)。

        参数: kv_caches:
            层名到 kv cache 的字典
        """

        self._lmcache_engine.register_kv_caches(kv_caches)

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        """
        Start loading the KV cache from the connector to vLLM's paged
        KV buffer. This is called from the forward context before the
        forward pass to enable async loading during model execution.

        Args:
            forward_context (ForwardContext): the forward context.
            **kwargs: additional arguments for the load operation
        """

        """
        开始把 KV cache 从 connector 加载到 vLLM 的 paged KV buffer。
        这个方法在 forward context 里、前向传播(forward pass)之前被调用,
        目的是让加载能在模型执行期间异步进行。

        参数:
            forward_context (ForwardContext): 前向上下文。
            **kwargs: 加载操作的额外参数
        """

        """
            [Note 笔记] 它只是发起加载,然后立刻返回,不等数据搬完。

            同步(阻塞)做法:            异步(LMCache 做法):
            start_load_kv()          start_load_kv()  ← 发起搬运,马上返回
                ↓ 傻等数据搬完...          ↓
                ↓ (GPU 闲着)         开始前向计算  ← 同时,KV 在后台悄悄搬
            数据到了                      ↓  
            开始前向计算             (计算和搬运重叠进行!)

            1. start_load_kv()      ← 你选的这个,发起异步加载
            2. 前向计算开始
            3. 算到某一层时,wait_for_layer_load()  ← 确保那层 KV 已到位
            4. 用这些 KV 继续算

        """
        self._lmcache_engine.start_load_kv(forward_context, **kwargs)

    def wait_for_layer_load(self, layer_name: str) -> None:
        """
        Block until the KV for a specific layer is loaded into vLLM's
        paged buffer. This is called from within attention layer to ensure
        async copying from start_load_kv is complete.

        This interface will be useful for layer-by-layer pipelining.

        Args:
            layer_name: the name of that layer
        """

        """
        阻塞,直到某个特定层的 KV 被加载进 vLLM 的 paged buffer。
        这个方法在注意力层(attention layer)内部被调用,用来确保
        start_load_kv 发起的异步拷贝已经完成。

        这个接口对逐层流水线(layer-by-layer pipelining)很有用。

        参数:
            layer_name: 那一层的名字
        """

        """
        时间轴 ──────────────────────────────────►

        搬运:  [搬第0层][搬第1层][搬第2层][搬第3层]...
        计算:         [算第0层][算第1层][算第2层]...
                    ↑
        算第0层前,wait 第0层搬完就行
        (第1、2、3层还在后台搬,不用等它们)
        

        前向开始前:
        start_load_kv()        ← 一次性发起"搬所有层的 KV"(异步,马上返回)

        前向计算中,逐层进行:
        算第0层 attention:
            wait_for_layer_load("layer0")  ← 确认第0层到位(可能瞬间返回)
            用第0层 KV 计算
        算第1层 attention:
            wait_for_layer_load("layer1")  ← 确认第1层到位
            用第1层 KV 计算
        ...
        """
        self._lmcache_engine.wait_for_layer_load(layer_name)


    # [疑问-待后面解答] KV 是不是都进 CPU?
    #   答: 不是。connector 只发起存/取动作, "存不存/存哪层"由下游决定:
    #     - 存不存 → cache_engine.store (阶段2)
    #     - 存哪个backend/多层路由 → storage_manager (阶段3)
    #     - CPU怎么存/满了怎么淘汰 → local_cpu_backend (阶段3)
    #   KV先在GPU算出, "进CPU"是额外留副本供复用, 取决于配置的backend+策略+容量。


    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs,
    ) -> None:
        """
        Start saving the a layer of KV cache from vLLM's paged buffer
        to the connector. This is called from within attention layer to
        enable async copying during execution.

        Args:
            layer_name (str): the name of the layer.
            kv_layer (torch.Tensor): the paged KV buffer of the current
                layer in vLLM.
            attn_metadata (AttentionMetadata): the attention metadata.
            **kwargs: additional arguments for the save operation.
        """

        """
        开始把一层 KV cache 从 vLLM 的 paged buffer 保存到 connector。
        这个方法在注意力层内部被调用,以便在执行期间进行异步拷贝。

        参数:
            layer_name (str): 层的名字。
            kv_layer (torch.Tensor): vLLM 中当前层的 paged KV 缓冲区。
            attn_metadata (AttentionMetadata): 注意力元数据。
            **kwargs: 保存操作的额外参数。
        [Note 笔记] ：
            当layer计算完之后，直接保存当前层的layer（异步的方式）
            也需要告诉atten_metadata，知道放在哪个slot
        """

        self._lmcache_engine.save_kv_layer(
            layer_name, kv_layer, attn_metadata, **kwargs
        )

    def wait_for_save(self):
        """
        Block until all the save operations is done. This is called
        as the forward context exits to ensure that the async saving
        from save_kv_layer is complete before finishing the forward.

        This prevents overwrites of paged KV buffer before saving done.
        """
        """
        阻塞,直到所有保存操作都完成。这个方法在 forward context 退出时
        被调用,用来确保 save_kv_layer 发起的异步保存在前向结束之前
        已经完成。

        这可以防止 paged KV buffer 在保存完成之前被覆盖。
        """
        self._lmcache_engine.wait_for_save()

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[Optional[set[str]], Optional[set[str]]]:
        """
        Notifies worker-side connector ids of requests that have
        finished generating tokens.

        Returns:
            ids of requests that have finished asynchronous transfer
            (requests that previously returned True from request_finished()),
            tuple of (sending/saving ids, recving/loading ids).
            The finished saves/sends req ids must belong to a set provided in a
            call to this method (this call or a prior one).
        """

        """
        通知 worker 侧 connector:哪些请求已经完成了 token 生成。

        返回:
            那些已经完成异步传输的请求 id
            (即之前从 request_finished() 返回过 True 的请求),
            返回一个元组: (发送/保存完成的 id, 接收/加载完成的 id)。
            这些完成的 save/send 请求 id 必须属于本方法某次调用
            (本次或之前某次)传入过的集合。

            [Note 笔记]： vLLM 想释放这个请求占用的 GPU block(腾地方给别人),但如果 KV 还没搬完就释放,数据就丢了。
            他的单位是request，而不是layer

        """
        return self._lmcache_engine.get_finished(finished_req_ids)

    def get_block_ids_with_load_errors(self) -> set[int]:
        """Return block IDs that failed to load during the last interval."""
        """返回在上一个时间间隔内加载失败的 block ID。"""
        return self._lmcache_engine.get_block_ids_with_load_errors()

    def shutdown(self):
        """
        Shutdown the connector. This is called when the worker process
        is shutting down to ensure that all the async operations are
        completed and the connector is cleaned up properly.
        """

        """
        关闭 connector。当 worker 进程正在关闭时调用,
        用来确保所有异步操作都已完成、connector 被正确清理。
        """
        return self._lmcache_engine.shutdown()

        """
            ① get_num_new_matched_tokens  ← 你现在读的:问"能复用多少 N?" ⭐起点
            ② (vLLM 据此分配 block, 规划本步)
            ③ update_state_after_alloc    ← 分配完告诉 LMCache
            ④ build_connector_meta        ← 打包"本步要 load 哪些 KV"的元数据
            ────────── 以上 scheduler-side,决定"复用计划" ──────────
            ⑤ start_load_kv               ← worker-side:按计划异步搬 KV(你已读过)
            ⑥ wait_for_layer_load         ← 逐层等到位(你已读过)
            ⑦ 前向计算,跳过那 N 个 token 的 prefill
        """

    # ==============================
    # Scheduler-side methods
    # ==============================
    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[Optional[int], bool]:
        """
        Get number of new tokens that can be loaded from the
        external KV cache beyond the num_computed_tokens.

        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally
                computed tokens for this request

        Returns:
            the number of tokens that can be loaded from the
            external KV cache beyond what is already computed.
        """

        """
        获取:在 num_computed_tokens 之外,还能从外部 KV cache 加载多少个新 token。

        参数:
            request (Request): 请求对象。
            num_computed_tokens (int): 这个请求本地已经计算过的 token 数量

        返回:
            在已经计算过的之外,还能从外部 KV cache 加载的 token 数量。
        """


        return self._lmcache_engine.get_num_new_matched_tokens(
            request, num_computed_tokens
        ), False

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        """
        Update KVConnector state after block allocation.
        """
        """
        在 block 分配之后,更新 KVConnector 的状态。
        """

        self._lmcache_engine.update_state_after_alloc(request, num_external_tokens)

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        """
        Build the connector metadata for this step.

        This function should NOT modify fields in the scheduler_output.
        Also, calling this function will reset the state of the connector.

        Args:
            scheduler_output (SchedulerOutput): the scheduler output object.
        """
        """
        为这一步(step)构建 connector 元数据。

        这个函数不应该修改 scheduler_output 里的字段。
        另外,调用这个函数会重置 connector 的状态。

        参数:
            scheduler_output (SchedulerOutput): 调度器输出对象。

        """
        return self._lmcache_engine.build_connector_meta(scheduler_output)

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        """
        Called when a request has finished, before its blocks are freed.

        Returns:
            True if the request is being saved/sent asynchronously and blocks
            should not be freed until the request_id is returned from
            get_finished().
            Optional KVTransferParams to be included in the request outputs
            returned by the engine.
        当一个请求结束、在它的 block 被释放之前调用。

        返回:
            True 表示这个请求正在异步地 保存/发送,它的 block 不应该被释放,
            直到这个 request_id 从 get_finished() 返回为止。
            以及可选的 KVTransferParams,会被包含在引擎返回的请求输出里。


        原因：vLLM当一个请求结束之后，会把所有的block 引用次数 - 1，如果是0就有可能被直接回收
        所以需要先问一下是不是LMcache需要先存储，就先冻结
        """
        return self._lmcache_engine.request_finished(request, block_ids)
