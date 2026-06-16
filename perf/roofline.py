from enum import Enum, auto

class RooflineBase():
    def get_roofline(self):
       return max(self.get_compute_cycles(), self.get_memory_cycles(), self.get_network_cycles())

    def get_compute_cycles(self):
        """
        Abstract method to calculate compute cycles.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement get_compute_cycles()")

    def get_memory_cycles(self):
        """
        Abstract method to calculate memory cycles.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement get_memory_cycles()")

    def get_network_cycles(self):
        """
        Abstract method to calculate network cycles.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement get_network_cycles()")

class SingleLayerTransformer(RooflineBase):
    class Sharding(Enum):
        ZERO = auto()
        TENSOR_PARALLEL = auto()
        CONTEXT_PARALLEL = auto()

    def __init__(self,
                 *,
                 batch_size : int,
                 seqlen: int,
                 model_dim: int,
                 ffn_dim: int,
                 num_q_heads: int,
                 num_kv_heads: int,
                 dtype_size: int):
        self.batch_size_ = batch_size
        self.seqlen_ = seqlen
        self.model_dim_ = model_dim
        self.ffn_dim_ = ffn_dim
        self.qh_ = num_q_heads
        self.kvh_ = num_kv_heads
        self.dtype_size_ = dtype_size

class Prefill(SingleLayerTransformer):
    def __init__(self,
                 *,
                 batch_size: int,
                 seqlen: int,
                 model_dim: int,
                 ffn_dim: int,
                 num_q_heads: int,
                 num_kv_heads: int,
                 dtype_size: int,
                 sharding_type: SingleLayerTransformer.Sharding):
        super().__init__(
            batch_size=batch_size,
            seqlen=seqlen,
            model_dim=model_dim,
            ffn_dim=ffn_dim,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            dtype_size=dtype_size
        )
        self.sharding_type_ = sharding_type
    

    def get_compute_cycles(self):
        pass

    def get_memory_cycles(self):
        # TODO: Implement prefill specific memory cycle calculation
        pass

    def get_network_cycles(self):
        # TODO: Implement prefill specific network cycle calculation
        pass
    
    
