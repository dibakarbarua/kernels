from dataclasses import dataclass
from typing import List, Optional
import heapq


@dataclass
class Request:
    request_id: int
    arrival_time: float
    prompt_len: int
    output_len: int
    generated: int = 0
    start_time: Optional[float] = None
    first_token_time: Optional[float] = None
    finish_time: Optional[float] = None

    @property
    def context_len(self) -> int:
        return self.prompt_len + self.generated

    @property
    def done(self) -> bool:
        return self.generated >= self.output_len


@dataclass
class Model:
    num_layers: int
    num_kv_heads: int
    head_dim: int
    dtype_bytes: int


@dataclass
class Hardware:
    hbm_bandwidth_GBps: float

    @property
    def bandwidth_Bps(self):
        return self.hbm_bandwidth_GBps * 1e9


class DecodeSimulator:
    def __init__(self, model: Model, hw: Hardware, max_batch_size: int):
        self.model = model
        self.hw = hw
        self.max_batch_size = max_batch_size

    def kv_bytes_for_request(self, req: Request) -> int:
        return (
            self.model.num_layers
            * 2
            * self.model.num_kv_heads
            * req.context_len
            * self.model.head_dim
            * self.model.dtype_bytes
        )

    def step_latency(self, active: List[Request]) -> float:
        batch = active[:self.max_batch_size]
        total_bytes = sum(self.kv_bytes_for_request(req) for req in batch)
        return total_bytes / self.hw.bandwidth_Bps

    def run(self, requests: List[Request]) -> dict:
        requests = sorted(requests, key=lambda r: r.arrival_time)

        t = 0.0
        i = 0
        active: List[Request] = []
        completed: List[Request] = []

        while i < len(requests) or active:
            # If no active requests, jump to next arrival.
            if not active and i < len(requests) and t < requests[i].arrival_time:
                t = requests[i].arrival_time

            # Admit newly arrived requests.
            while i < len(requests) and requests[i].arrival_time <= t:
                req = requests[i]
                req.start_time = t
                active.append(req)
                i += 1

            if not active:
                continue

            # Choose batch.
            batch = active[:self.max_batch_size]

            dt = self.step_latency(batch)

            # Decode one token for each request in batch.
            for req in batch:
                req.generated += 1

                if req.first_token_time is None:
                    req.first_token_time = t + dt

                if req.done:
                    req.finish_time = t + dt

            t += dt

            # Remove completed.
            still_active = []
            for req in active:
                if req.done:
                    completed.append(req)
                else:
                    still_active.append(req)
            active = still_active

        total_output_tokens = sum(r.output_len for r in completed)
        total_time = max(r.finish_time for r in completed if r.finish_time is not None) \
            - min(r.arrival_time for r in completed)

        avg_latency = sum((r.finish_time - r.arrival_time) \
                          for r in completed if r.finish_time is not None) / len(completed)
        avg_ttft = sum((r.first_token_time - r.arrival_time) for r in completed if r.first_token_time is not None) / len(completed)

        return {
            "num_requests": len(completed),
            "total_output_tokens": total_output_tokens,
            "total_time_s": total_time,
            "tokens_per_second": total_output_tokens / total_time if total_time > 0 else 0.0,
            "average_latency_s": avg_latency,
            "average_ttft_s": avg_ttft,
        }