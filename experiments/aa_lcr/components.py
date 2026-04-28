import os
import asyncio
from typing import Dict, Tuple, Optional
import csv

class Object:
    def __init__(
        self,
        name: str,
        num_context_tokens: int,
        rate: float = 0.0,
        rate_score: dict | None = None,
    ):
        """
        name: 唯一标识
        size: 原始大小 (GB)
        rate: 压缩率 (0=不压缩，0.5=压50%)
        context_tokens: 全量 tokens，不考虑压缩
        rate_score: 字典 {rate: score}，记录不同压缩率下的得分
        """
        self.name = name
        self.size = kv_cache_calculator(num_context_tokens)
        self.rate = rate
        self.device = None  # 'cpu' / 'ssd' / 'remote'
        self.rate_score = rate_score if rate_score is not None else {}
        self.num_context_tokens = num_context_tokens

    @property
    def effective_size(self) -> float:
        """考虑压缩之后真实占用大小 (GB)。"""
        return self.size * (1.0 - self.rate)


class StorageManager:
    def __init__(self):
        # 容量 (GB)
        self.cpu_size = float(os.environ.get("CPU_SIZE"))
        self.ssd_size = float(os.environ.get("SSD_SIZE"))
        # remote 无限容量，这里只做统计，不做约束
        self.cpu_used = 0.0
        self.ssd_used = 0.0
        self.remote_used = 0.0

        self.cpu_objects = {}     # name -> Object
        self.ssd_objects = {}     # name -> Object
        self.remote_objects = {}  # name -> Object
        self.storages = {}        # 全局表：name -> Object

        self._lock = asyncio.Lock()

        alpha_str = os.environ.get("ALPHA")
        self._alpha: float = float(alpha_str)
        ttft_csv = os.environ.get("TTFT_CSV")
        self._ttft_params = self._load_ttft_params(ttft_csv)

    # ---------- utility helpers (dummy for now) ----------
    
    @staticmethod
    def _load_ttft_params(csv_path: str) -> Dict[Optional[str], Tuple[float, float]]:
        """
        从 CSV (device,A,B) 读出 TTFT 参数，返回:
            {device: (A, B), ...}
        其中 device 可以是 "cpu" / "ssd" / "remote" / None
        """
        params: Dict[Optional[str], Tuple[float, float]] = {}

        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                dev_raw = (row.get('device') or row.get('\ufeffdevice') or '').strip()
                a_raw = (row.get("A") or "").strip()
                b_raw = (row.get("B") or "").strip()

                # 跳过空行或不完整行（你给的 CSV 最后一行就是空的）
                if not a_raw or not b_raw:
                    continue

                device: Optional[str] = dev_raw if dev_raw not in ("", "None") else None
                try:
                    A = float(a_raw)
                    B = float(b_raw)
                except ValueError:
                    raise ValueError(f"Invalid A/B in TTFT CSV: {row!r}")

                params[device] = (A, B)

        if not params:
            raise RuntimeError(f"No valid rows found in TTFT CSV: {csv_path}")

        return params

    def _candidate_rates(self, obj: Object) -> list[float]:
        """
        Return a sorted list of candidate compression rates for this object.
        We restrict to [0.0, 0.9] as per your description.
        """
        rates = [r for r in obj.rate_score.keys() if 0.0 <= r <= 0.9]
        if rates:
            return sorted(set(rates))

    def _utility(self, obj: Object, device: str | None, rate: float) -> float:
        """
        Dummy utility function.

        You can replace this later with your real utility that depends on
        num_context_tokens, rate_score, device, etc.

        Currently:
          - base_score = obj.rate_score.get(rate, 0.0)
          - device_weight: cpu > ssd > remote > None
        """
        score = obj.rate_score.get(rate)
        A, B = self._ttft_params.get(device)
        TTFT = A * obj.num_context_tokens * (1 - rate) + B
        utility = self._alpha * score - TTFT
        return utility

    def _best_unconstrained_config(self, obj: Object) -> tuple[str, float]:
        """
        For this object, choose (device, rate) that maximizes utility,
        ignoring capacity.

        Devices considered: cpu / ssd / remote.
        (We treat None as eviction-only, not something you "aim" for.)
        """
        best_device = None
        best_rate = 0.0
        best_u = self._utility(obj, best_device, best_rate)

        rates = self._candidate_rates(obj)
        for device in ("cpu", "ssd", "remote"):
            for r in rates:
                u = self._utility(obj, device, r)
                if u > best_u:
                    best_u = u
                    best_device = device
                    best_rate = r

        return best_device, best_rate

    # ---------- low-level: 只做操作 & 更新 size，不负责策略 ----------

    def _evict_object(self, obj: Object):
        """
        Completely remove obj from storages and usage accounting.
        This corresponds to device=None, rate=0 in your description.
        """
        name = obj.name
        # current effective size
        eff = obj.effective_size
        dev = obj.device

        if dev == "cpu":
            self.cpu_used -= eff
            self.cpu_objects.pop(name, None)
        elif dev == "ssd":
            self.ssd_used -= eff
            self.ssd_objects.pop(name, None)
        elif dev == "remote":
            self.remote_used -= eff
            self.remote_objects.pop(name, None)

        self.storages.pop(name, None)
    
    def insert(self, obj: Object, device: str, rate: float | None = None):
        """
        低层操作：把一个“新对象”放到指定 device，并更新 used 容量。
        - 不做容量检查
        - 不做策略
        - 如果传入 rate，则覆盖 obj.rate

        注意：如果同名对象已经存在，会报错；更新请用 change()。
        """
        if obj.name in self.storages:
            raise ValueError(f"Object {obj.name} already exists; use change() instead.")

        # Compress the object
        if rate is not None:
            obj.rate = rate

        obj.device = device
        eff = obj.effective_size

        if device == "cpu":
            self.cpu_objects[obj.name] = obj
            self.cpu_used += eff
        elif device == "ssd":
            self.ssd_objects[obj.name] = obj
            self.ssd_used += eff
        elif device == "remote":
            self.remote_objects[obj.name] = obj
            self.remote_used += eff
        else:
            raise ValueError(f"Unknown device: {device!r}")

        self.storages[obj.name] = obj
        return obj

    def change(self, name: str, device: str | None = None, rate: float | None = None):
        """
        低层操作：更新已有对象的 device / rate，并同步更新各层 used 容量。
        - device / rate 任意一个可以为 None，表示“不改这个字段”
        - 不做容量检查，不做策略。

        返回更新后的 Object；如果找不到该 name 则返回 None。
        """
        obj = self.storages.get(name)
        if obj is None:
            return None

        # 旧占用
        old_eff = obj.effective_size
        old_device = obj.device

        # 从旧层移除占用
        if old_device == "cpu":
            self.cpu_used -= old_eff
            self.cpu_objects.pop(name, None)
        elif old_device == "ssd":
            self.ssd_used -= old_eff
            self.ssd_objects.pop(name, None)
        elif old_device == "remote":
            self.remote_used -= old_eff
            self.remote_objects.pop(name, None)

        # 更新属性 (contains compression)
        if rate is not None:
            obj.rate = rate
        if device is None:
            device = old_device
        obj.device = device

        # 新占用
        new_eff = obj.effective_size

        # 挂到新层
        if device == "cpu":
            self.cpu_objects[name] = obj
            self.cpu_used += new_eff
        elif device == "ssd":
            self.ssd_objects[name] = obj
            self.ssd_used += new_eff
        elif device == "remote":
            self.remote_objects[name] = obj
            self.remote_used += new_eff
        else:
            raise ValueError(f"Unknown device: {device!r}")

        return obj

    # ---------- rebalancing for one device (CPU / SSD) ----------

    def _rebalance_cpu(self):
        """
        Repeatedly apply the operation with minimal marginal utility loss
        until CPU usage <= cpu_size.

        Operations considered for each CPU object:
          - Increase compression rate on CPU (rate_old -> rate_new >= rate_old).
          - Move to SSD with rate in [0, 0.9].
          - Move to remote with rate in [0, 0.9].
          - Evict (device=None, rate=0).
        """
        while self.cpu_used > self.cpu_size and self.cpu_objects:
            best_op = None
            best_marginal = float("inf")  # we minimize this

            # Snapshot, because we'll mutate inside the loop
            cpu_objs = list(self.cpu_objects.values())

            for obj in cpu_objs:
                old_rate = obj.rate
                old_device = "cpu"
                old_eff = obj.effective_size

                old_u = self._utility(obj, old_device, old_rate)
                rates = self._candidate_rates(obj)

                # 1) More compression on CPU (rate_old -> rate_new >= rate_old)
                for r_new in rates:
                    if r_new <= old_rate:
                        continue
                    new_u = self._utility(obj, "cpu", r_new)
                    util_loss = old_u - new_u
                    saved = obj.size * (r_new - old_rate)  # CPU saved size
                    marginal = util_loss / saved
                    if marginal < best_marginal:
                        best_marginal = marginal
                        best_op = ("compress_cpu", obj, r_new)

                # 2) Move to SSD with any allowed rate (0-0.9)
                for r_new in rates:
                    new_u = self._utility(obj, "ssd", r_new)
                    util_loss = old_u - new_u
                    saved = old_eff  # we free all current CPU usage
                    marginal = util_loss / saved
                    if marginal < best_marginal:
                        best_marginal = marginal
                        best_op = ("move_to_ssd", obj, r_new)

                # 3) Move to remote with any allowed rate (0-0.9)
                for r_new in rates:
                    new_u = self._utility(obj, "remote", r_new)
                    util_loss = old_u - new_u
                    saved = old_eff
                    marginal = util_loss / saved
                    if marginal < best_marginal:
                        best_marginal = marginal
                        best_op = ("move_to_remote", obj, r_new)

                # 4) Evict (device=None, rate=0)
                new_u = self._utility(obj, None, 0.0)
                util_loss = old_u - new_u
                saved = old_eff
                marginal = util_loss / saved
                if marginal < best_marginal:
                    best_marginal = marginal
                    best_op = ("evict", obj, 0.0)

            if best_op is None:
                # No operation can save capacity; break to avoid infinite loop.
                break

            kind, obj, r_new = best_op

            if kind == "compress_cpu":
                print(f"Compressing object {obj.name} on CPU to rate {r_new}")
                self.change(obj.name, device="cpu", rate=r_new)
            elif kind == "move_to_ssd":
                # During CPU rebalance, we allow SSD to temporarily exceed capacity.
                print(f"Moving object {obj.name} from CPU to SSD at rate {r_new}")
                self.change(obj.name, device="ssd", rate=r_new)
            elif kind == "move_to_remote":
                print(f"Moving object {obj.name} from CPU to remote at rate {r_new}")
                self.change(obj.name, device="remote", rate=r_new)
            elif kind == "evict":
                print(f"Evicting object {obj.name} from CPU")
                self._evict_object(obj)

    def _rebalance_ssd(self):
        """
        Repeatedly apply the operation with minimal marginal utility loss
        until SSD usage <= ssd_size.

        Operations considered for each SSD object:
          - Increase compression rate on SSD (rate_old -> rate_new >= rate_old).
          - Move to remote with rate in [0, 0.9].
          - Evict (device=None, rate=0).

        We do NOT move things back to CPU here to avoid re-violating CPU capacity.
        """
        while self.ssd_used > self.ssd_size and self.ssd_objects:
            best_op = None
            best_marginal = float("inf")

            ssd_objs = list(self.ssd_objects.values())

            for obj in ssd_objs:
                old_rate = obj.rate
                old_device = "ssd"
                old_eff = obj.effective_size

                old_u = self._utility(obj, old_device, old_rate)
                rates = self._candidate_rates(obj)

                # 1) More compression on SSD
                for r_new in rates:
                    if r_new <= old_rate:
                        continue
                    new_u = self._utility(obj, "ssd", r_new)
                    util_loss = old_u - new_u
                    saved = obj.size * (r_new - old_rate)  # SSD saved size
                    marginal = util_loss / saved
                    if marginal < best_marginal:
                        best_marginal = marginal
                        best_op = ("compress_ssd", obj, r_new)

                # 2) Move to remote with any allowed rate (0-0.9)
                for r_new in rates:
                    new_u = self._utility(obj, "remote", r_new)
                    util_loss = old_u - new_u
                    saved = old_eff
                    marginal = util_loss / saved
                    if marginal < best_marginal:
                        best_marginal = marginal
                        best_op = ("move_to_remote", obj, r_new)

                # 3) Evict
                new_u = self._utility(obj, None, 0.0)
                util_loss = old_u - new_u
                saved = old_eff
                marginal = util_loss / saved
                if marginal < best_marginal:
                    best_marginal = marginal
                    best_op = ("evict", obj, 0.0)

            if best_op is None:
                break

            kind, obj, r_new = best_op

            if kind == "compress_ssd":
                self.change(obj.name, device="ssd", rate=r_new)
            elif kind == "move_to_remote":
                self.change(obj.name, device="remote", rate=r_new)
            elif kind == "evict":
                self._evict_object(obj)
    
    # ---------- policy ----------

    def policy(self, obj: Object):
        """
        1) For this object, compute its best (device, rate) by utility.
        2) Insert it to that device ignoring capacity.
        3) If CPU exceeds capacity, repeatedly choose the operation
           with minimal marginal utility loss on CPU (compress / move / evict)
           until CPU is within limit.
        4) If SSD exceeds capacity, do the same on SSD.
        """
        # Step 1: best unconstrained choice for this object
        best_device, best_rate = self._best_unconstrained_config(obj)

        print(f"Placing object {obj.name}: device={best_device}, rate={best_rate}")

        # Step 2: insert ignoring capacity
        if best_device:
            self.insert(obj, device=best_device, rate=best_rate)

        # Step 3: rebalance CPU if needed
        if self.cpu_used > self.cpu_size:
            self._rebalance_cpu()

        # Step 4: rebalance SSD if needed
        if self.ssd_used > self.ssd_size:
            self._rebalance_ssd()

    # ---------- existence check + record + create ----------

    async def get_or_create(
        self,
        name: str,
        num_context_tokens: int,
        rate_score: dict,
    ) -> tuple[bool, tuple[float, str]]:
        """
        原子操作：
          - 如果已存在：hit=True，返回已有 (rate, device)
          - 如果不存在：创建、放置，hit=False，返回新 (rate, device)
        """
        async with self._lock:
            obj = self.storages.get(name)
            if obj is not None:
                # 已经有了，直接返回
                return True, (obj.rate, obj.device)

            # MISS -> 创建并放置
            size = kv_cache_calculator(num_context_tokens)
            obj = Object(
                name=name,
                size=size,
                num_context_tokens=num_context_tokens,
                rate=0.0,
                rate_score=rate_score,
            )
            self.policy(obj)  # policy 里调用 insert

            return False, (obj.rate, obj.device)

def kv_cache_calculator(length: int) -> float:
    model = os.getenv("MODEL", "")
    if model in (
        "meta-llama/Llama-3.1-8B-Instruct",
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "mistralai/Mistral-7B-Instruct-v0.3"
    ):
        gb_per_token = 0.1221 / 1000  # GB
    elif model == "Qwen/Qwen2.5-14B-Instruct":
        gb_per_token = 0.1831 / 1000  # GB
    elif model == "lmsys/longchat-7b-v1.5-32k":
        gb_per_token = 0.4883 / 1000  # GB
    elif model == "Qwen/Qwen3-30B-A3B-Instruct-2507":
        gb_per_token = 0.0915 / 1000  # GB
    elif model == "openai/gpt-oss-120b":
        # 18 full-attention layers (the other 18 are sliding_window=128 and
        # contribute a constant ~4.5 MB independent of context length, ignored).
        # Per token (bf16): 18 layers * 2 (K,V) * 8 kv_heads * 64 head_dim * 2 B
        # = 36 864 B = 3.43e-5 GB/token = 0.0343 / 1000 GB/token.
        gb_per_token = 0.0343 / 1000  # GB
    total_gb = length * gb_per_token
    return total_gb
