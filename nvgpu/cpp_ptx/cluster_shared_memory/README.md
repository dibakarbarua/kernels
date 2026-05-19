# Cluster Shared Memory Tutorial

This example reduces a 1D `float` tensor with a deliberately small PTX-first
kernel. It is meant for reading, not peak performance.

Cluster shared memory is still declared as ordinary CUDA shared memory:

```cuda
__shared__ float cluster_slot;
```

The cluster part begins when we convert that CUDA C++ pointer to a PTX shared
address and map it to a CTA rank inside the cluster:

```cuda
unsigned local_slot = ptx_shared_u32addr(&cluster_slot);
unsigned peer_slot = ptx_mapa_shared_cluster(local_slot, peer_rank);
```

Then the actual cluster-visible read/write instructions are:

```ptx
st.shared::cluster.f32 [peer_slot], value;
ld.shared::cluster.f32 value, [peer_slot];
```

Build on Hopper / SM90 or newer:

```bash
bash ../build_and_run.sh cluster_shared_memory --sm 90
```

The important PTX instructions in the source are:

- `cvta.to.shared` to convert a CUDA C++ pointer into shared address space
- `mapa.shared::cluster` to map a shared address to a peer CTA rank
- `st.shared::cluster` and `ld.shared::cluster` for cluster-visible access
- `barrier.cluster.arrive` and `barrier.cluster.wait` for peer visibility
- `red.global.add.f32` to combine one value per cluster into the final output
