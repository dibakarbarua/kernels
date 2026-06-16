from dataclasses import dataclass, field
from typing import Dict, List
import math

@dataclass
class RequestState:
    request_id: str
    length: int
    blocks: List[int] = field(default_factory=list)


class KVBlockAllocator:
    def __init__(self, num_blocks: int, block_size: int):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.free_blocks = list(range(num_blocks))
        self.requests: Dict[str, RequestState] = {}

    def _blocks_needed(self, num_tokens: int) -> int:
        return math.ceil(num_tokens / self.block_size)

    def allocate(self, request_id: str, prompt_len: int) -> bool:
        needed = self._blocks_needed(prompt_len)

        if needed > len(self.free_blocks):
            return False

        blocks = [self.free_blocks.pop() for _ in range(needed)]
        self.requests[request_id] = RequestState(
            request_id=request_id,
            length=prompt_len,
            blocks=blocks,
        )
        return True

    def append_token(self, request_id: str) -> bool:
        req = self.requests[request_id]

        # This part is important!!
        old_needed = self._blocks_needed(req.length)
        new_needed = self._blocks_needed(req.length + 1)

        if new_needed > old_needed:
            if not self.free_blocks:
                return False
            req.blocks.append(self.free_blocks.pop())

        req.length += 1
        return True

    def free(self, request_id: str):
        req = self.requests.pop(request_id)
        self.free_blocks.extend(req.blocks)

    def stats(self):
        allocated_blocks = self.num_blocks - len(self.free_blocks)
        live_tokens = sum(req.length for req in self.requests.values())
        capacity_tokens = allocated_blocks * self.block_size
        waste_tokens = capacity_tokens - live_tokens

        return {
            "num_live_requests": len(self.requests),
            "allocated_blocks": allocated_blocks,
            "free_blocks": len(self.free_blocks),
            "live_tokens": live_tokens,
            "capacity_tokens": capacity_tokens,
            "internal_fragmentation_tokens": waste_tokens,
            "utilization": live_tokens / capacity_tokens if capacity_tokens else 1.0,
        }