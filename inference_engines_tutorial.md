# A Robust Mental Model of an Inference Serving Engine

This document describes a generic large-language-model inference serving
system. It borrows architectural ideas visible in engines such as vLLM,
TensorRT-LLM, and SGLang, but it is deliberately implementation-independent.

The central idea is:

> An inference serving engine is a token-level, memory-aware scheduling system
> wrapped around a model executor.

The model forward pass is important, but it is only one block. A production
system must also control admission, queueing, memory, fairness, cancellation,
streaming, failures, and observability.

A thorough treatment has been done by Aleksa Gordic at https://www.aleksagordic.com/blog/vllm

## 1. The system at three scales

It helps to reason about three nested scales.

### Fleet scale

A fleet routes requests among replicas:

```text
clients
   |
edge gateway / load balancer
   |
request router
   |
   +---- model replica A: engine + GPU group
   +---- model replica B: engine + GPU group
   +---- model replica C: engine + GPU group
```

A replica may use one GPU, or a group of GPUs connected through tensor,
pipeline, expert, or context parallelism. Data parallelism usually means
running several replicas and routing requests among them.

### Request scale

One request has a lifecycle:

```text
receive
  -> authenticate and validate
  -> render prompt and tokenize
  -> estimate cost and admit
  -> wait in the engine queue
  -> prefill, possibly in chunks
  -> sample and stream the first output token
  -> alternate waiting and decode execution
  -> stop, cancel, fail, or time out
  -> release live resources
```

### Iteration scale

The engine itself runs a much faster loop:

```text
inspect all waiting and running requests
  -> choose a token budget for this iteration
  -> allocate any KV blocks needed now
  -> build a mixed execution batch
  -> run one model step
  -> sample tokens and update request states
  -> emit stream events
  -> retire completed work
  -> repeat
```

This is continuous batching. A request is not normally assigned to one static
batch for its entire lifetime. The batch is reconstructed at each iteration as
requests arrive, finish, block, or are cancelled.

## 2. Overall block architecture

```text
                              CONTROL PLANE
            model loading, config, autoscaling, health, metrics
                                     |
                                     v
+----------+   +----------+   +-------------+   +-------------+
| Client   |-->| Gateway  |-->| API server  |-->| Preparation |
| HTTP/RPC |<--| auth/RL  |<--| stream I/O  |<--| + validate  |
+----------+   +----------+   +-------------+   +-------------+
                                                        |
                                                        v
                                               +----------------+
                                               | Admission and  |
                                               | replica routing|
                                               +----------------+
                                                        |
                                                        v
                    ONE MODEL REPLICA / ENGINE
+-----------------------------------------------------------------------+
|  +---------------+       +----------------+                           |
|  | Waiting queue |------>| Scheduler      |<----- cancellations       |
|  +---------------+       | token budgets  |                           |
|                          | priority/fair  |                           |
|                          +-------+--------+                           |
|                                  |                                    |
|                 +----------------+----------------+                   |
|                 v                                 v                   |
|       +------------------+              +-------------------+         |
|       | KV cache manager |              | Batch assembler   |         |
|       | blocks/prefixes  |------------->| tensors/metadata  |         |
|       +------------------+              +---------+---------+         |
|                                                  |                    |
|                                                  v                    |
|                                      +-----------------------+        |
|                                      | Distributed executor  |        |
|                                      | GPU workers/model run |        |
|                                      +-----------+-----------+        |
|                                                  |                    |
|                                                  v                    |
|                                      +-----------------------+        |
|                                      | Logits + sampling     |        |
|                                      | constraints + stops   |        |
|                                      +-----------+-----------+        |
|                                                  |                    |
|                                      +-----------+-----------+        |
|                                      | state updates / output|        |
|                                      +-----------------------+        |
+-----------------------------------------------------------------------+
                                                        |
                                                        v
                                             detokenize and stream
```

Here `RL` means rate limiting.

In a real deployment these blocks need not be separate processes. The useful
mental boundary is ownership:

- The API layer owns protocol and connection state.
- The admission layer owns overload policy and user-level policy.
- The engine scheduler owns runnable request state.
- The KV manager owns logical-to-physical cache mappings.
- The executor owns device execution.
- The output layer owns ordered delivery and backpressure.

## 3. Persistent state versus live execution state

The most important boundary is between application state and engine state.

| Scope | Typical state | Lifetime | Usual owner |
|---|---|---|---|
| User/account | identity, plan, quotas, billing, safety policy | months or years | application/control plane |
| Conversation | messages, tool results, attachments, summary | minutes to years | application database |
| Request | prompt, generation settings, deadline, trace ID | one API call | API layer and engine |
| Sequence | token IDs, generated count, KV block table, RNG state | prefill through retirement | engine |
| Connection | socket, stream cursor, send buffer, disconnect flag | one HTTP/RPC stream | API server |
| Replica | model weights, tokenizer, KV pool, CUDA graphs/workspaces | process lifetime | engine/executor |
| Iteration | selected sequences, token counts, tensor metadata | one scheduler step | scheduler/model runner |

### What per-user state belongs inside the engine?

Usually very little:

- A user or tenant ID used for accounting and fairness.
- A priority class or service-level objective.
- Counters such as queued requests, active sequences, or token debt.
- Possibly a routing-affinity hint.
- Possibly a cache-isolation or cache-salt identifier.

The engine should not normally be the source of truth for a chat transcript,
user profile, tool history, or durable memory. Keeping those in the engine
would bind application correctness to one replica's process lifetime.

### What happens when a conversation becomes inactive?

There are two distinct events:

1. The current generation request finishes or is cancelled.
2. The logical conversation receives no new turns for some period.

For a conventional engine, event 1 causes the live sequence to be retired and
its KV-cache lease to be released. Event 2 is mostly irrelevant to the engine,
because the conversation record is application state.

On the next turn, the application renders the relevant conversation history
into a new prompt. The engine tokenizes and prefills it again unless a reusable
prefix is still available.

An optional prefix cache may retain unreferenced KV blocks after a request
finishes. Those blocks are:

- An optimization, not durable conversation storage.
- Evictable whenever memory pressure requires it.
- Reusable only when the effective token prefix and model context match.
- Subject to isolation rules for adapters, tenants, multimodal inputs, and
  other inputs that affect model computation.

Therefore:

```text
conversation continuity != guaranteed KV-cache continuity
```

Some specialized systems support explicit pause/resume, KV offload, or remote
KV stores. Treat that as a separate state-management feature, not the baseline
meaning of "conversation."

## 4. A useful request state machine

```text
RECEIVED
   |
   v
PREPARING --invalid--> REJECTED
   |
   v
ADMISSION_WAIT --overload/quota--> REJECTED
   |
   v
QUEUED --deadline/cancel--> CANCELLED
   |
   v
PREFILLING <----+
   |            |
   |       PREEMPTED/PAUSED
   v            ^
DECODING -------+
   |
   +--stop/max tokens--> FINISHING --> FINISHED
   +--disconnect/cancel-> CANCELLED
   +--executor error----> FAILED
```

`PREFILLING` and `DECODING` do not mean continuously executing on a GPU. A
sequence may spend most wall-clock time between iterations waiting for its
next scheduling turn.

Preemption means the engine temporarily removes a request from active
execution to reclaim a scarce resource, usually KV capacity. Recovery might
recompute a prefix, reload offloaded KV, or restore a checkpoint. It is
expensive and should be visible in metrics.

## 5. Python-style data abstractions

These types describe ownership, not a recommended implementation.

```python
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import AsyncIterator, Protocol, Sequence

TokenId = int
RequestId = str
TenantId = str


class Phase(Enum):
    RECEIVED = auto()
    PREPARING = auto()
    QUEUED = auto()
    PREFILLING = auto()
    DECODING = auto()
    PREEMPTED = auto()
    FINISHING = auto()
    FINISHED = auto()
    CANCELLED = auto()
    REJECTED = auto()
    FAILED = auto()


@dataclass(frozen=True)
class GenerationParameters:
    max_new_tokens: int
    temperature: float
    top_p: float
    top_k: int | None
    stop_token_ids: frozenset[TokenId]
    stop_strings: tuple[str, ...]
    seed: int | None
    return_logprobs: bool = False # log-probability allows lower precision stores (exp(-x) will require more precision)


@dataclass(frozen=True)
class RequestEnvelope:
    request_id: RequestId
    tenant_id: TenantId
    model_id: str
    messages_or_prompt: object
    parameters: GenerationParameters
    deadline_monotonic_s: float | None
    priority_class: str
    adapter_id: str | None
    trace_context: dict[str, str]


@dataclass(frozen=True)
class PreparedRequest:
    envelope: RequestEnvelope
    prompt_token_ids: tuple[TokenId, ...]
    prompt_fingerprint: str
    constraint: "CompiledConstraint | None"
    estimated_cost: "CostEstimate"


@dataclass(frozen=True)
class CostEstimate:
    prompt_tokens: int
    max_output_tokens: int
    initial_kv_blocks: int
    maximum_kv_blocks: int
    prefill_work_units: float
    decode_work_units: float


@dataclass
class SamplingState:
    rng_state: object
    generated_token_ids: list[TokenId] = field(default_factory=list)
    constraint_state: object | None = None
    stop_matcher_state: object | None = None


@dataclass
class OutputState:
    next_event_index: int = 0
    unflushed_token_ids: list[TokenId] = field(default_factory=list)
    client_connected: bool = True
    stream_backpressured: bool = False


@dataclass
class RequestTimestamps:
    received_at: float
    prepared_at: float | None = None
    admitted_at: float | None = None
    first_scheduled_at: float | None = None
    prefill_done_at: float | None = None
    first_token_at: float | None = None
    finished_at: float | None = None


@dataclass
class SequenceState:
    request: PreparedRequest
    phase: Phase
    prompt_cursor: int
    sampling: SamplingState
    output: OutputState
    kv_lease: "KVLease | None"
    timestamps: RequestTimestamps
    finish_reason: str | None = None
```

Notice what is absent: there is no durable `Conversation` object in
`SequenceState`. The rendered conversation has become prompt tokens for this
one request.

## 6. Front door and request preparation

The front door does more work than merely parsing JSON:

```python
class RequestValidator(Protocol):
    def validate_protocol(self, request: RequestEnvelope) -> None: ...
    def validate_model_limits(self, request: RequestEnvelope) -> None: ...


class PromptRenderer(Protocol):
    def render(self, messages_or_prompt: object, model_id: str) -> str: ...


class Tokenizer(Protocol):
    async def encode(self, text: str) -> Sequence[TokenId]: ...
    async def decode_incrementally(
        self, token_ids: Sequence[TokenId]
    ) -> str: ...


class ConstraintCompiler(Protocol):
    async def compile(self, schema_or_grammar: object) -> "CompiledConstraint": ...


class RequestPreparer(Protocol):
    async def prepare(self, request: RequestEnvelope) -> PreparedRequest: ...
```

Preparation can consume substantial non-GPU resources:

- An accepted network connection and HTTP/RPC parser state.
- Request-body and attachment memory.
- Authentication, authorization, quota, and safety-service calls.
- CPU time for chat-template rendering and tokenization.
- CPU or accelerator time for image/audio preprocessing.
- Memory for token IDs and multimodal embeddings.
- CPU time and memory for compiling a JSON schema or grammar.
- Queue entries, cancellation handles, trace spans, and stream buffers.
- Adapter/model metadata lookup.

This is why overload control must exist before the GPU scheduler. A system can
exhaust sockets, CPU, RAM, or queue memory while its GPUs still look healthy.

### Kernel-level interpretation of the pre-GPU stages

If your normal starting point is "an operator has already been dispatched,"
the pre-GPU path is everything required to turn remote application data into
the small integer and metadata tensors consumed by that operator.

#### Connections are live operating-system and protocol state

A connection is not an abstract user identity. It is a live communication
channel, usually a TCP socket carrying HTTP, HTTP/2, WebSocket, or RPC traffic.
One streaming request may keep that channel open for the entire generation.

State associated with a connection commonly includes:

- A file descriptor and kernel socket object.
- Kernel receive and send buffers.
- TCP congestion, retransmission, and flow-control state.
- TLS encryption state and record buffers for HTTPS.
- HTTP parser state, headers, and request-body buffers.
- An asynchronous task, coroutine, or event-loop registration.
- A response queue and bytes waiting for the client to read them.
- A disconnect and cancellation signal tied to the engine request.

Most of this lives in kernel memory or process CPU RAM. It consumes file
descriptors, memory, event-loop work, encryption CPU time, and network
bandwidth. A client that reads slowly can fill the server's output buffers even
if token generation itself is fast.

The connection and the model request also have different lifetimes. HTTP/2 may
multiplex several requests over one connection, and a keep-alive connection
may remain after one inference request finishes.

#### Tokenization is not an embedding GEMM

There are two operations that are easy to conflate:

```text
text bytes --tokenizer--> integer token IDs
integer token IDs --embedding-table lookup--> hidden vectors
```

The first is normally a CPU string-processing algorithm. Depending on the
tokenizer, it performs operations such as Unicode normalization, regular
expression or rule-based pre-tokenization, byte encoding, trie lookup, BPE
merge selection, WordPiece matching, or unigram-model segmentation. The work
is irregular, branch-heavy, variable-length, and driven by lookup tables. It
is not naturally expressed as a dense GEMM.

For example:

```text
"unbelievable"
    -> UTF-8 bytes / normalized text
    -> candidate substrings or byte pieces
    -> vocabulary and merge-table lookups
    -> [token_id_0, token_id_1, token_id_2]
```

The second operation happens inside the model on the accelerator:

```python
hidden_0 = embedding_weight[input_token_ids]
```

That is generally a gather from the embedding table, not a GEMM either. The
later transformer projections are GEMMs.

CPU tokenization is common because request strings arrive in CPU memory,
individual prompts may be small, tokenizer control flow is irregular, and
moving raw strings plus complex tokenizer tables to the GPU can cost more than
it saves. It can still be a serious front-end bottleneck at high request rates
or with very long prompts. Implementations therefore use native-code
tokenizers, CPU thread pools, caching, batching, or specialized GPU
tokenization where the workload justifies it.

The output is usually an `int32` or `int64` CPU array. Only the selected token
IDs and associated batch metadata need to be copied to the GPU before model
execution.

#### Multimodal preprocessing spans CPU and GPU work

For an image request, the incoming payload is often compressed JPEG, PNG, or
WebP bytes. Those bytes are not directly suitable input to a vision encoder.
A representative path is:

```text
network bytes in CPU RAM
  -> decode compressed image
  -> inspect dimensions and validate limits
  -> resize / crop / color conversion
  -> normalize and convert layout/dtype
  -> pixel tensor
  -> H2D transfer
  -> GPU vision encoder
  -> visual embeddings consumed by the language model
```

Audio may require container parsing, decoding, resampling, channel conversion,
and feature extraction. Video adds frame selection and potentially many image
decodes.

Image/audio decoding and shape-dependent transformations are often CPU work,
although GPU decode and preprocessing libraries exist. The expensive neural
encoder is commonly a GPU workload. Some systems run it on the same GPU as the
LLM; others have a separate encoder pool.

The encoder output is usually a tensor of embeddings or model-specific visual
tokens, not text-token IDs. It may remain in GPU memory for LLM prefill, be
copied between devices, or be cached if the same media is reused.

## 7. Admission control and routing

Admission asks, "Should this work enter the bounded system?" Scheduling asks,
"Which admitted tokens should execute in the next iteration?"

```python
class AdmissionKind(Enum):
    ADMIT = auto()
    QUEUE = auto()
    REJECT = auto()
    RETRY_ELSEWHERE = auto()


@dataclass(frozen=True)
class AdmissionDecision:
    kind: AdmissionKind
    reason: str
    retry_after_s: float | None = None
    effective_priority: float = 0.0


@dataclass(frozen=True)
class ReplicaSnapshot:
    healthy: bool
    queued_requests: int
    running_sequences: int
    queued_prefill_tokens: int
    kv_blocks_free: int
    kv_blocks_total: int
    recent_prefill_tokens_per_s: float
    recent_decode_tokens_per_s: float
    estimated_queue_delay_s: float


class AdmissionController(Protocol):
    def decide(
        self,
        request: PreparedRequest,
        tenant: "TenantPolicySnapshot",
        fleet: Sequence[ReplicaSnapshot],
    ) -> AdmissionDecision: ...


class ReplicaRouter(Protocol):
    def choose_replica(
        self,
        request: PreparedRequest,
        candidates: Sequence[ReplicaSnapshot],
    ) -> str: ...
```

Useful admission inputs include:

- Per-tenant request and token rate limits.
- Number of that tenant's queued and active requests.
- Prompt length and requested maximum output length.
- Deadline and predicted queueing delay.
- Model, adapter, quantization, and hardware compatibility.
- Free KV capacity and probability of a prefix-cache hit.
- Current mix of prefill-heavy and decode-heavy work.
- Priority class and fairness debt.
- Whether another replica can serve the request more cheaply.

Admission need not reserve the request's maximum possible KV memory. Doing so
is simple but wastes capacity because many generations stop early. Engines
often allocate KV blocks incrementally while retaining safety headroom.

### A bounded-system admission sketch

```python
def decide(request, tenant, fleet):
    if tenant.requests_in_window >= tenant.request_limit:
        return reject("tenant request-rate limit", retry_after_s=1.0)

    if tenant.active_sequences >= tenant.max_active_sequences:
        return reject("tenant concurrency limit", retry_after_s=0.5)

    viable = [r for r in fleet if r.healthy and can_run(request, r)]
    if not viable:
        return retry_elsewhere("no healthy compatible replica")

    best = min(viable, key=lambda r: predicted_completion_time(request, r))

    if deadline_will_be_missed(request, best):
        return reject("deadline infeasible")

    if best.queued_requests >= MAX_BOUNDED_QUEUE:
        return reject("replica queue full", retry_after_s=estimate_retry(best))

    return admit(priority=weighted_fair_priority(request, tenant))
```

Do not use an unbounded queue as the overload policy. At sustained arrival
rates above service capacity, it merely converts explicit rejection into
arbitrarily large latency and eventual memory exhaustion.

## 8. What "at capacity" actually means

Capacity is multidimensional. A system can be at capacity in one of several
ways:

| Bottleneck | Observable shape | Consequence |
|---|---|---|
| Front-end | high connections, CPU, RAM, event-loop lag | requests wait before tokenization |
| Tokenizer/preprocessor | CPU pool saturated | growing preparation delay |
| Prefill compute | long prompt queue, low decode pressure | TTFT grows |
| Decode bandwidth | many live sequences, high GPU memory traffic | TPOT grows |
| KV capacity | few free blocks, preemptions/evictions | admission stalls or recomputation |
| Network/output | send buffers and disconnects grow | generated tokens cannot drain |
| Distributed fabric | collective or KV-transfer latency grows | all sequences in a batch slow |

The correct response depends on the scarce resource:

- Reject with `429` for tenant quota/rate limits.
- Reject with `503` or an overload status when the service has no feasible
  capacity.
- Return a retry hint when retrying later is meaningful.
- Queue only inside a strict bound and preferably only if the deadline remains
  feasible.
- Route to another healthy replica or model tier.
- Stop admitting large prompts before KV exhaustion.
- Preempt lower-priority work only when the policy justifies its latency cost.
- Autoscale when the workload and startup time make that useful.

At saturation, queueing delay rises nonlinearly. Even before nominal throughput
is exceeded, burstiness and variable request sizes create head-of-line effects.
Little's Law is a useful sanity check:

```text
number of requests in system = arrival rate * average time in system
L = lambda * W
```

If throughput is flat while `L` grows, `W` must be growing. That is overload,
even if GPU utilization looks impressive.

## 9. KV-cache memory is the engine's dynamic heap

Model weights are mostly static after loading. KV cache is dynamic and grows
with every live sequence.

For a conventional transformer, an approximate full-replica KV cost per token
is:

```text
KV bytes per token
  = number_of_layers
    * 2                    # key and value
    * number_of_KV_heads
    * head_dimension
    * bytes_per_element
```

The local amount on each worker depends on how the model and KV heads are
partitioned. Multi-query and grouped-query attention reduce
`number_of_KV_heads`. KV quantization reduces `bytes_per_element`.

With paged KV allocation:

```text
blocks_for_sequence
  = ceil(tokens_whose_KV_is_stored / tokens_per_block)
```

Internal fragmentation is bounded roughly by the unused portion of the last
block rather than by preallocating every sequence to its maximum length.

### KV abstractions

```python
@dataclass(frozen=True)
class PhysicalBlock:
    block_id: int


@dataclass
class KVLease:
    request_id: RequestId
    logical_to_physical: list[PhysicalBlock]
    tokens_materialized: int
    shared_prefix_blocks: int


@dataclass(frozen=True)
class PrefixKey:
    model_fingerprint: str
    adapter_fingerprint: str | None
    tenant_cache_salt: str | None
    token_block_hashes: tuple[str, ...]
    extra_input_fingerprint: str | None


class KVCacheManager(Protocol):
    def lookup_prefix(self, request: PreparedRequest) -> "PrefixMatch": ...

    def create_lease(
        self, request_id: RequestId, prefix: "PrefixMatch"
    ) -> KVLease: ...

    def can_append(self, lease: KVLease, token_count: int) -> bool: ...

    def allocate_append(self, lease: KVLease, token_count: int) -> None: ...

    def publish_reusable_prefix(self, lease: KVLease) -> None: ...

    def release(self, lease: KVLease) -> None: ...

    def evict_unreferenced(self, blocks_needed: int) -> int: ...
```

Three counts must not be conflated:

- **Allocated:** physical memory currently assigned to block objects.
- **Referenced:** blocks needed by a live sequence.
- **Cached:** unreferenced or shared blocks retained for possible prefix reuse.

Releasing a request may reduce its reference counts without immediately
zeroing or physically deallocating the entire GPU memory pool. The engine can
recycle those blocks for another request.

## 10. Prefix reuse

Prefix reuse avoids recomputing KV entries for token blocks that are identical
under the same effective model computation.

Conceptually:

```python
prefix = kv_manager.lookup_prefix(request)
lease = kv_manager.create_lease(request.request_id, prefix)
state.prompt_cursor = prefix.matched_tokens

# Only the unmatched suffix needs prefill.
remaining_prompt = request.prompt_token_ids[state.prompt_cursor:]
```

A safe cache key may need to include more than token IDs:

- Exact model and weight version.
- Adapter or LoRA identity.
- Positional-encoding or context configuration.
- Multimodal inputs or embeddings.
- Cache isolation salt or tenant boundary.
- Any other input that changes hidden states.

Prefix caching improves prefill and therefore often TTFT. It does not remove
the autoregressive work needed to produce new output tokens.

### Exactly what constitutes a shared prefix?

A prefix is first a property of the **ordered model input**, not of a tensor
shape or a natural-language topic.

Suppose request A starts with:

```text
[system tokens] [document tokens] [user-question-A tokens]
```

and request B starts with:

```text
[system tokens] [document tokens] [user-question-B tokens]
```

Their shared prefix is the longest identical initial token sequence:

```text
[system tokens] [document tokens]
```

If the engine uses 16-token KV blocks, it will usually share only the complete
matching blocks. A partially matching final block may need recomputation or
copy-on-write handling, depending on the cache design.

For causal self-attention, the hidden state for token position `i` depends only
on positions `0..i`, never on a future suffix. Therefore, if two requests have
the same effective inputs through position `i`, the K and V vectors already
computed at those positions can be reused:

```text
for every layer l:
    K_l[0 : shared_length]
    V_l[0 : shared_length]
```

This remains true even if the requests have different suffixes. Each suffix
will create new queries that attend to the same shared K/V history.

### A realistic cross-user example

Consider a company serving an internal assistant. Every request begins with:

1. A 2,000-token system policy.
2. A 1,000-token description of available tools and their JSON schemas.
3. A 6,000-token employee handbook.
4. The individual employee's question.

The effective prompts might look like:

```text
User Alice:
    [company system policy: 2,000 tokens]
    [tool definitions:       1,000 tokens]
    [employee handbook:      6,000 tokens]
    ["How many vacation days can I carry over?"]

User Bob:
    [company system policy: 2,000 tokens]
    [tool definitions:       1,000 tokens]
    [employee handbook:      6,000 tokens]
    ["What is the parental-leave policy?"]
```

If those first 9,000 tokens are byte-for-byte tokenized identically and use
the same model configuration, the engine can reuse their per-layer KV blocks:

```text
Alice: [========== shared 9,000-token KV ==========][Alice suffix KV]
Bob:   [========== same physical KV blocks =======][Bob suffix KV]
```

The execution timeline is:

```text
Alice arrives first:
    prefill 9,000 shared tokens + Alice's suffix
    retain or publish eligible KV blocks in the prefix cache

Bob arrives later:
    prefix lookup matches 9,000 tokens
    increment reference counts on those physical KV blocks
    prefill only Bob's unmatched question suffix
```

Bob does not receive Alice's question, output, hidden state, or attention
scores. Both sequences merely reference identical immutable K/V history for
the content they truly have in common. Their block tables diverge at the first
different token.

This pattern occurs with:

- A shared long system prompt.
- Repeated few-shot examples.
- Identical tool definitions.
- A common RAG document placed before each user's question.
- Requests against the same code file or document.
- Parallel samples or branches generated from one prompt.

The practical detail is **prompt ordering**. Prefix caching only helps a
contiguous initial region. This ordering is cache-friendly:

```text
[shared policy][shared document][user-specific question]
```

This ordering destroys most cross-request prefix reuse:

```text
[user-specific metadata][shared policy][shared document]
```

because Alice's and Bob's prompts differ near token zero. Production prompt
builders may deliberately place stable shared material first, provided doing
so preserves the model's intended semantics and security policy.

### Is sharing usually across different users?

It can be, but there are several important reuse populations:

| Reuse population | Example | Typical value |
|---|---|---|
| Same request | beam search, parallel samples, speculative branches | very high and immediate |
| Same conversation | previous chat history reused on the next turn | high when routed to a warm cache |
| Same application, different users | shared system prompt, tools, documents, examples | often the largest aggregate opportunity |
| Same tenant | many employees querying tenant-specific policy or code | high with a useful isolation boundary |
| Different unrelated tenants | provider-wide public system material | technically possible, often restricted |

Cross-user reuse is attractive because application-owned prefixes repeat at
large scale. However, it is not automatically safe or desirable. A serving
provider may salt cache keys by tenant:

```python
cache_key = hash(
    tenant_cache_salt,
    model_version,
    adapter_id,
    token_block,
)
```

With the same salt, Alice and Bob can share if they belong to the permitted
cache domain. With different salts, identical token blocks deliberately do
not match.

Isolation may be required because of:

- Side-channel concerns: cache-hit timing could reveal that another user
  recently submitted a particular prefix.
- Different data-retention or residency policies.
- Tenant-specific adapters, authorization, or hidden prompt material.
- Accounting rules that require cache resources to remain tenant-scoped.
- Operational simplicity when routing and eviction are tenant-aware.

The clean mental model is:

> Prefix reuse is based on model-input equality, while the set of users allowed
> to share that equality is defined by the cache-isolation policy.

### What is actually cached?

The baseline reusable payload is the per-layer **key and value tensors**:

```python
shared_payload = {
    layer: (
        key_cache[layer, shared_positions, local_kv_heads, :],
        value_cache[layer, shared_positions, local_kv_heads, :],
    )
    for layer in model_layers
}
```

It is generally **not**:

- The Q projection for every old token.
- The attention score matrix.
- The softmax probabilities.
- All intermediate hidden activations.
- The final vocabulary logits for every prefix position.

The reason follows directly from the next decode operator:

```text
q_new = Q_projection(hidden_new)
attention_new = Attention(q_new, K_history, V_history)
```

The new token supplies a new query. Old queries are no longer required. Old
attention scores are also useless because scores are query-dependent:

```text
score(new, old_position) = q_new dot k_old
```

Modern fused attention kernels often do not materialize the full score or
probability matrix even during the original computation; they tile through K
and V and maintain online softmax statistics.

Hidden activations are normally released layer by layer because retaining them
for every token would add another large persistent-memory burden. K and V are
the particular intermediates required by all future causal-attention steps, so
they are the persistent state.

Vocabulary logits are produced only after the final hidden state passes
through the LM head. They are useful for choosing the next token at that
specific sequence boundary, but they are not the history consumed by future
attention. A specialized cache could retain terminal hidden states or logits,
but that is auxiliary state rather than the usual meaning of a KV prefix
cache. An engine with a full KV hit may retain suitable terminal state or
recompute a small boundary portion to obtain next-token logits.

### Equality means equal effective computation

Identical human-readable text is not always sufficient. Reuse is valid only if
the cached positions would produce the same K/V bits, or sufficiently
equivalent values under the engine's correctness policy.

The cache identity may include:

```python
CacheIdentity(
    token_ids=exact_token_prefix,
    positions=effective_positions,
    model_weights=model_version,
    adapter=adapter_or_lora_id,
    attention_config=rope_and_window_configuration,
    multimodal_inputs=media_or_embedding_fingerprint,
    tenant_salt=isolation_boundary,
)
```

Common reasons two apparently similar prompts cannot share include:

- Different chat templates or whitespace produce different token IDs.
- Different model weights, quantization, or LoRA adapters change K/V.
- Different positional treatment changes RoPE-applied keys or hidden states.
- Different image embeddings occupy positions inside the prefix.
- A sliding-window or other attention policy changes which history is valid.
- Security policy deliberately prevents cross-tenant cache reuse.

Prefix sharing is therefore best understood as:

> Two sequences reference the same immutable physical KV blocks for their
> longest model-equivalent initial token range.

After the shared point, each sequence gets its own logical blocks. Reference
counts keep shared blocks alive, and writes to a partially shared block require
copy-on-write or a block-boundary split.

## 11. Scheduler: the heart of the engine

The scheduler deals in token budgets, sequence slots, KV blocks, priorities,
and deadlines.

```python
@dataclass(frozen=True)
class WorkItem:
    request_id: RequestId
    phase: Phase
    input_token_count: int
    kv_tokens_to_append: int
    position_start: int


@dataclass(frozen=True)
class SchedulePlan:
    work: tuple[WorkItem, ...]
    preempt: tuple[RequestId, ...]
    retire: tuple[RequestId, ...]
    total_scheduled_tokens: int


class Scheduler(Protocol):
    def enqueue(self, state: SequenceState) -> None: ...

    def cancel(self, request_id: RequestId) -> None: ...

    def plan(
        self,
        now: float,
        max_batched_tokens: int,
        max_running_sequences: int,
        kv: KVCacheManager,
    ) -> SchedulePlan: ...

    def apply_results(self, results: Sequence["SequenceStepResult"]) -> None: ...
```

A simple policy might:

1. Remove cancelled or expired requests.
2. Give already-running decode sequences enough service to protect TPOT.
3. Add chunks from prefill requests until the token budget is exhausted.
4. Admit queued requests when sequence slots and KV headroom permit.
5. Use weighted fairness so one tenant cannot monopolize the engine.
6. Preempt only if policy and recovery cost make it worthwhile.

### Simplified scheduling pseudo-code

```python
def plan(now, token_budget, sequence_budget, kv):
    plan = SchedulePlanBuilder()

    for seq in expired_or_cancelled(now):
        plan.retire(seq.request_id)

    # Protect interactive decode latency.
    for seq in fair_order(runnable_decode_sequences()):
        if plan.sequence_count == sequence_budget:
            break
        if token_budget < 1:
            break
        if ensure_kv_for_decode(seq, kv):
            plan.add_decode(seq, input_tokens=1)
            token_budget -= 1

    # Fill the remaining budget with prompt chunks.
    for seq in fair_order(runnable_prefill_sequences()):
        if plan.sequence_count == sequence_budget:
            break
        chunk = min(seq.remaining_prompt_tokens, token_budget, PREFILL_CHUNK)
        chunk = shrink_to_available_kv(chunk, seq, kv)
        if chunk > 0:
            plan.add_prefill(seq, input_tokens=chunk)
            token_budget -= chunk

    return plan.build()
```

Real policies must also handle cache hits, speculative tokens, beam groups,
structured-output constraints, adapters, multimodal encoders, distributed
placement, and preemption. The abstraction remains the same: select feasible
token work under resource and policy constraints.

### Why chunk prefill?

A very long prefill can otherwise occupy a large iteration and delay decode
tokens for all currently streaming users. Chunking lets the engine interleave
compute-heavy prompt processing with latency-sensitive decode steps.

The tradeoff is that smaller chunks may reduce prefill efficiency, while very
large chunks may harm inter-token latency. The scheduler is choosing a point
on the throughput-versus-latency curve.

## 12. Batch assembly and execution

The scheduler's logical plan must be converted into device-ready metadata:

```python
@dataclass(frozen=True)
class ExecutionBatch:
    input_token_ids: "DeviceTensor"
    positions: "DeviceTensor"
    sequence_offsets: "DeviceTensor"
    kv_block_tables: "DeviceTensor"
    context_lengths: "DeviceTensor"
    phase_metadata: object


@dataclass(frozen=True)
class ModelStepOutput:
    logits_or_selected_logits: "DeviceTensor"
    updated_kv_metadata: object


class BatchAssembler(Protocol):
    def build(
        self,
        plan: SchedulePlan,
        states: dict[RequestId, SequenceState],
    ) -> ExecutionBatch: ...


class ModelExecutor(Protocol):
    async def execute(self, batch: ExecutionBatch) -> ModelStepOutput: ...
```

For a distributed replica, `execute` fans the same logical batch out to a
group of workers. The workers run their model partitions and perform required
collectives. The engine should regard that worker group as one failure and
scheduling domain unless the architecture explicitly supports finer recovery.

Static device resources usually include:

- Model weights.
- Communication buffers.
- Kernel workspaces.
- Captured graphs or compiled execution variants.
- The preallocated physical KV pool.

Dynamic resources include:

- This iteration's input and metadata buffers.
- Temporary activations and attention workspaces.
- New KV blocks assigned to live sequences.
- Logits or reduced sampling outputs.

### Concrete storage ledger for one request

The word "request memory" can be misleading because some physical allocations
are process-wide pools while only their logical ownership is per request. For
example, an engine may reserve a 40 GiB KV arena at startup. Request R does not
call `cudaMalloc` for 2 GiB; it receives references to blocks inside that
arena.

The following describes a conventional server with discrete CPU and GPU
memory. Unified-memory systems and device-resident schedulers can move these
boundaries.

A conceptual residency model is:

```python
class MemoryTier(Enum):
    STORAGE = auto()
    HOST_PAGEABLE = auto()
    HOST_PINNED = auto()
    DEVICE_HBM = auto()
    REMOTE_KV = auto()


@dataclass(frozen=True)
class BufferRef:
    tier: MemoryTier
    dtype: str
    shape: tuple[int, ...]
    byte_count: int
    owner: str              # replica, request, sequence, or iteration
    lifetime: str


@dataclass
class RequestResidency:
    raw_request: BufferRef | None
    prompt_token_ids_host: BufferRef
    media_host: BufferRef | None
    kv_blocks_device: list[BufferRef]
    generated_token_ids_host: BufferRef
    output_bytes_host: BufferRef
```

`HOST_PINNED` is commonly a shared staging tier for asynchronous DMA rather
than the long-term home of every request. Implementations may copy from normal
pageable request memory into reusable pinned batch buffers immediately before
H2D transfer.

#### Replica-wide storage, not charged anew to each request

| Object | Typical location | Lifetime | How it gets there |
|---|---|---|---|
| Model checkpoint | local/remote storage | deployment artifact | downloaded before or during startup |
| Model weights | GPU HBM, sharded across workers | replica lifetime | storage -> CPU buffers -> H2D, or direct-storage path |
| Tokenizer vocabulary and merge tables | CPU RAM | process lifetime | loaded from model files |
| KV physical block pool | GPU HBM | replica lifetime | reserved during engine initialization |
| Kernel workspaces | GPU HBM | process or execution lifetime | allocated during warmup/runtime |
| CUDA graphs/compiled variants | GPU memory plus CPU metadata | process lifetime | produced during warmup/compilation |
| NCCL/collective buffers | GPU HBM and communication resources | process lifetime | initialized with distributed workers |
| Adapter cache | CPU RAM and/or GPU HBM | until eviction | loaded on demand or at startup |

The model weights dominate static storage. A model with `P` parameters stored
at `b` bytes per parameter needs approximately:

```text
weight bytes across replica ~= P * b
```

before accounting for scales, metadata, padding, duplicated embeddings, and
runtime workspaces. Tensor parallelism shards much of this storage among GPUs;
it does not make the total model representation disappear.

#### Per-request CPU-side storage

| Object | Typical representation | Approximate scaling | Lifetime |
|---|---|---|---|
| Protocol state | headers, IDs, options | usually KiB-scale | receive through response close |
| Raw prompt/messages | UTF-8 or JSON bytes | input byte length | preparation, sometimes request lifetime |
| Raw media | compressed byte buffer | uploaded media size | validation/preprocessing |
| Decoded media | pixels/audio samples | decoded dimensions | preprocessing, possibly cache lifetime |
| Prompt token IDs | `int32[]` or `int64[]` | 4 or 8 bytes per token | request lifetime |
| Generated token history | token ID vector | 4 or 8 bytes per output token | generation lifetime |
| Scheduler state | phase, counters, deadline, priority | small fixed metadata | engine lifetime of request |
| KV block table mirror | block IDs/reference metadata | proportional to KV blocks | prefill through retirement |
| Sampling metadata | temperature, seed, penalties | small, plus token history | generation lifetime |
| Constraint state | FSM/parser state and tables | schema-dependent | generation lifetime |
| Detokenizer state | byte suffix and decoded text cursor | output-dependent | streaming lifetime |
| Output buffers | pending events/text bytes | bounded by policy | until client drains or is cancelled |

CPU token arrays are tiny compared with KV. Four thousand `int32` prompt
tokens occupy only about 16 KiB. The original UTF-8/JSON request and media can
be larger, but for text-only LLM requests the GPU-side KV state generally
dominates dynamic per-request storage.

The CPU often keeps the full token history even though a copy or representation
also exists through GPU KV state. Token IDs are needed for stop handling,
penalties, output construction, tracing, recomputation after preemption, and
possibly retries. KV is not a reversible replacement for the token sequence.

#### Per-request persistent GPU-side storage

| Object | Typical representation | Approximate scaling | Notes |
|---|---|---|---|
| KV cache | paged K/V tensors for every local layer | context tokens * KV bytes/token | usually the dominant dynamic allocation |
| Multimodal embeddings | encoder output tensor | visual/audio token count * hidden size | may be retained through prefill or cached |
| Device sampling state | RNG state, penalty/constraint metadata | small or history-dependent | implementation-dependent |
| Device token history | token IDs or compact state | input + output tokens | optional; useful for GPU sampling |
| Per-sequence device metadata | lengths, slots, block-table references | small | often packed across all active sequences |

There is usually no permanent per-request allocation for all Q tensors,
attention scores, MLP activations, or vocabulary logits. Those belong to one
model iteration and are recycled.

The KV attribution for one logical sequence is approximately:

```text
logical KV bytes
  = stored_context_tokens
    * layers
    * 2
    * local_or_global_KV_heads
    * head_dimension
    * bytes_per_KV_element
```

For total storage across the replica, use the model's global KV-head count.
For one tensor-parallel worker, use the KV heads stored by that worker and
account for any replication required by the parallelism scheme.

Paged allocation rounds this to blocks:

```text
physical bytes attributed to request
  ~= ceil(stored_context_tokens / block_tokens)
     * bytes_per_KV_block
```

Shared prefix blocks complicate accounting. Their physical bytes exist once,
while several requests hold references. A capacity ledger can charge them
once globally, divide cost among references, or charge each tenant logically;
those choices answer different billing and fairness questions.

#### Per-iteration transient GPU storage

Once the scheduler selects work, the request contributes rows or tokens to a
batch rather than owning an isolated execution arena.

| Object | Shape intuition | Lifetime |
|---|---|---|
| Input token IDs | `[scheduled_tokens]` | this iteration |
| Positions/slot mapping | `[scheduled_tokens]` | this iteration |
| Context lengths/offsets | `[scheduled_sequences]` | this iteration |
| Device block tables | packed block IDs for selected sequences | iteration or persistent packed table |
| Input hidden states | `[scheduled_tokens, hidden_size]` | layer pipeline, recycled |
| Q projection | `[scheduled_tokens, local_query_heads, head_dim]` | current attention layer |
| New K/V projection | `[scheduled_tokens, local_kv_heads, head_dim]` | written into persistent KV |
| Attention output | `[scheduled_tokens, hidden_size]` | current layer |
| MLP intermediates | model-dependent expanded width | current layer |
| Collective buffers | local shard communication | current layer/operation |
| Final hidden rows | usually one relevant row per decode sequence | until LM head |
| Logits | relevant rows by vocabulary shard or full vocabulary | until sampling |
| Sampling scratch | top-k candidates, probabilities, scans | until token selection |

Optimized execution reuses activation buffers between layers and fuses
operators. Flash-style attention avoids allocating a full
`[query_tokens, context_tokens]` score matrix in HBM. Consequently, "attention
scores for the request" are generally not a persistent object and may never
exist as a complete tensor.

Transient memory is governed by the scheduler's current token budget and
batch composition. It is more accurate to attribute it to an iteration than
to one request, although large prefill chunks clearly contribute more of it.

### CPU/GPU transfer timeline

Here is a conventional end-to-end data path:

```text
1. Network interface
     -> kernel socket receive buffers in host memory

2. API server
     -> request bytes / JSON / media buffers in process CPU memory

3. CPU preparation
     -> rendered text
     -> tokenizer produces CPU token-ID array
     -> optional media decode produces CPU pixel/audio tensor

4. Engine queue
     -> CPU request metadata and token IDs remain resident
     -> no model execution has happened yet

5. Scheduler selects a prefill chunk or decode token
     -> CPU constructs packed token IDs, positions, lengths, slot mappings,
        and KV block-table updates

6. Batch staging
     -> metadata is commonly packed into pinned host buffers
     -> asynchronous H2D copies place it in GPU buffers
     -> raw text is not copied to the model

7. GPU model execution
     -> token IDs gather embedding rows
     -> layer kernels produce transient Q/K/V and activations
     -> new K/V values are written directly into assigned HBM KV blocks
     -> the final hidden row passes through the LM-head GEMM
     -> logits remain in HBM

8. Sampling
     -> GPU kernels usually apply logit transforms and select token IDs
     -> a tiny result, such as token ID and requested logprob, is copied D2H
        or delivered through an IPC queue

9. CPU output path
     -> selected token ID updates request state
     -> incremental detokenizer converts IDs to text bytes
     -> protocol layer frames an event
     -> bytes enter user-space and kernel send buffers
     -> network sends them to the client

10. Next decode iteration
     -> selected token becomes the next model input
     -> its ID may be staged H2D again, or remain device-resident in a more
        GPU-driven engine
     -> existing KV normally stays in HBM and is read in place

11. Retirement
     -> CPU request metadata and stream state are freed
     -> KV block references are decremented
     -> physical KV blocks return to the free/cache pool
```

The important bandwidth observation is:

> The full KV history is normally not copied CPU-to-GPU on every token. It
> stays resident in HBM; each decode step adds one token's K/V per layer and
> reads the existing history from HBM.

Optional KV offload changes this. A preempted or paused sequence may move KV
blocks from GPU HBM to pinned CPU RAM, another GPU, or remote storage. Resuming
it then requires transfer or recomputation before attention can use that
history. Disaggregated prefill/decode similarly transfers KV between worker
pools, often through GPU-to-GPU networking or RDMA rather than staging through
ordinary pageable CPU memory.

### A numerical KV example

Consider a transformer with:

```text
32 layers
8 KV heads
128 values per KV head
BF16 KV elements = 2 bytes
```

Then:

```text
KV bytes/token
  = 32 * 2 * 8 * 128 * 2
  = 131,072 bytes
  = 128 KiB per token across the replica
```

A 4,096-token context therefore represents:

```text
4,096 * 128 KiB = 512 MiB of logical KV
```

before block rounding and allocator metadata. By comparison, its CPU
`int32` token array is only:

```text
4,096 * 4 bytes = 16 KiB
```

If KV heads are evenly sharded over eight workers, each worker would store
roughly one eighth of that KV, subject to the model's actual parallel layout.
This comparison explains why serving schedulers reason so intensely about KV
capacity while prompt-token buffers receive much less attention.

## 13. Exact prefill and decode semantics

Suppose the prompt tokens are:

```text
x0, x1, ..., xN
```

Prefill processes these prompt tokens, writes their keys and values into the
KV cache, and produces logits for the first output token:

```text
prefill(prompt) -> logits(y0)
sample logits(y0) -> y0
stream y0
```

The next decode iteration processes the newly sampled token using the previous
KV state:

```text
decode(y0, KV(prompt)) -> KV(prompt + y0), logits(y1)
sample logits(y1) -> y1
stream y1
```

Then:

```text
decode(y1, KV(prompt + y0)) -> logits(y2)
...
```

One subtle consequence is that a token can be sampled and streamed before its
own KV entry is needed. Its KV entry is materialized when that token is fed
into the next decode step. If the token ends the sequence, no subsequent
decode may be necessary.

Implementations can fuse or rearrange these operations, but the dependency
chain is invariant.

## 14. Sampling, constraints, and stopping

Sampling is stateful per sequence:

```python
@dataclass(frozen=True)
class SampleResult:
    token_id: TokenId
    logprob: float | None
    next_constraint_state: object | None


class Sampler(Protocol):
    def sample(
        self,
        logits: "LogitsView",
        parameters: GenerationParameters,
        state: SamplingState,
    ) -> SampleResult: ...


class StopDetector(Protocol):
    def observe(
        self,
        token: SampleResult,
        state: SequenceState,
    ) -> "StopDecision": ...
```

### LM head first, sampling second

The operator order is:

```text
final transformer hidden state
    |
    v
LM-head projection: hidden @ W_vocab
    |
    v
vocabulary logits
    |
    v
logit processors / constraints / temperature
    |
    v
top-k, top-p, softmax or equivalent selection procedure
    |
    v
sampled integer token ID
```

So top-k sampling does **not** happen before the LM-head matmul. The LM head is
what creates one score for each vocabulary item:

```python
logits = final_hidden @ lm_head_weight.T
next_token_id = sample(process_logits(logits))
```

For decode, the relevant hidden-state input is often conceptually
`[active_sequences, hidden_size]`, and the LM head produces
`[active_sequences, vocabulary_size]` logits, possibly partitioned across
tensor-parallel ranks.

Sampling may include:

1. Applying token bans, grammar masks, repetition/frequency penalties, and
   other logit processors.
2. Dividing by temperature.
3. Finding top-k candidates, top-p candidates, or another candidate set.
4. Computing normalized probabilities as needed.
5. Drawing from the distribution with per-sequence RNG state.
6. Returning a token ID and optional logprob.

This is commonly a GPU operation in modern high-throughput engines. Copying
the entire vocabulary-sized logit row to the CPU every decode step would add
substantial D2H traffic and synchronization. GPU sampling keeps logits in HBM
and returns only a small result.

CPU sampling is still architecturally possible and may appear in simple
runtimes, unusual custom processors, or low-throughput implementations. Also,
some constraint logic may run on the CPU and produce a compact allowed-token
set or mask consumed by GPU kernels. "Sampling" names the semantic stage; it
does not require one fixed device placement.

With vocabulary-parallel LM heads, an engine can gather logits, perform a
distributed top-k/reduction, or otherwise coordinate sampling across shards.
That is an executor detail hidden behind the sampler interface.

Per-request sampling state can include:

- Random-number-generator state.
- Temperature, top-k, top-p, min-p, or penalties.
- Previously generated tokens used by repetition penalties.
- Logit processors and banned-token masks.
- A finite-state-machine state for JSON, regex, or grammar constraints.
- Stop-token and stop-string matcher state.
- Beam or speculative-decoding state.

Stopping can be caused by:

- End-of-sequence or another configured stop token.
- A matched stop string.
- Maximum generated tokens.
- Maximum model context.
- Deadline expiration.
- Explicit cancellation or client disconnect.
- Safety or policy intervention.
- Executor failure.

Stop-string detection may require holding back a small text suffix so bytes
that later become part of a stop string are not prematurely emitted.

## 15. Streaming and backpressure

Streaming is an asynchronous subsystem, not a `print()` after every token:

```python
@dataclass(frozen=True)
class StreamEvent:
    request_id: RequestId
    index: int
    text_delta: str
    token_id: TokenId | None
    logprob: float | None
    finish_reason: str | None


class OutputChannel(Protocol):
    async def send(self, event: StreamEvent) -> None: ...
    def is_connected(self) -> bool: ...
    def buffered_bytes(self) -> int: ...
    async def close(self) -> None: ...
```

Sampling and streaming are completely different layers:

```text
sampling:  logits -> token ID
streaming: token ID -> text bytes -> protocol frame -> network client
```

After the GPU selects token ID `12345`, the output path generally:

1. Copies that small result to CPU-visible memory, unless a device-side engine
   retains it and reports it asynchronously.
2. Feeds the ID to an incremental tokenizer decoder.
3. Produces zero or more text bytes. A token need not map to one complete
   Unicode character, so the detokenizer may buffer a partial byte sequence.
4. Applies stop-string withholding or output formatting.
5. Packages a Server-Sent Event, HTTP chunk, WebSocket message, or RPC frame.
6. Writes bytes to a user-space send buffer and then the kernel socket.

Streaming is therefore mostly CPU, operating-system, and network work. It can
be delayed, batched, backpressured, or disconnected independently of GPU
sampling. An engine may sample several tokens before the client receives any
of them, even though interactive services try to keep that delay small.

The engine must preserve per-request output order even though different
requests complete each iteration in a changing batch.

A slow client creates backpressure. The service needs a policy:

- Buffer up to a strict per-request limit.
- Temporarily stop scheduling that sequence.
- Cancel it after a timeout.
- Disconnect it rather than allowing unbounded memory growth.

Continuing expensive GPU generation for a disconnected client is wasted work.
Cancellation should propagate from the protocol layer to the scheduler, but a
small cancellation delay is normal if a device iteration is already in flight.

## 16. The end-to-end engine loop

```python
class InferenceEngine:
    def __init__(
        self,
        scheduler: Scheduler,
        kv: KVCacheManager,
        assembler: BatchAssembler,
        executor: ModelExecutor,
        sampler: Sampler,
        stop_detector: StopDetector,
        outputs: "OutputRegistry",
    ):
        self.scheduler = scheduler
        self.kv = kv
        self.assembler = assembler
        self.executor = executor
        self.sampler = sampler
        self.stop_detector = stop_detector
        self.outputs = outputs
        self.states: dict[RequestId, SequenceState] = {}

    def submit(self, request: PreparedRequest) -> AsyncIterator[StreamEvent]:
        state = make_initial_sequence_state(request)
        prefix = self.kv.lookup_prefix(request)
        state.kv_lease = self.kv.create_lease(request.envelope.request_id, prefix)
        state.prompt_cursor = prefix.matched_tokens
        state.phase = Phase.QUEUED
        self.states[request.envelope.request_id] = state
        self.scheduler.enqueue(state)
        return self.outputs.subscribe(request.envelope.request_id)

    async def run_forever(self) -> None:
        while True:
            self._propagate_disconnects_and_deadlines()

            plan = self.scheduler.plan(
                now=monotonic_time(),
                max_batched_tokens=MAX_BATCHED_TOKENS,
                max_running_sequences=MAX_RUNNING_SEQUENCES,
                kv=self.kv,
            )

            self._retire(plan.retire)
            self._preempt(plan.preempt)

            if not plan.work:
                await wait_for_new_work_or_timer()
                continue

            batch = self.assembler.build(plan, self.states)
            model_output = await self.executor.execute(batch)
            step_results = self._sample_update_and_detect_stop(
                plan, model_output
            )

            self.scheduler.apply_results(step_results)
            await self._publish_ready_output(step_results)
            self._retire_finished(step_results)

    def _retire_one(self, state: SequenceState) -> None:
        if state.kv_lease is not None:
            self.kv.publish_reusable_prefix(state.kv_lease)
            self.kv.release(state.kv_lease)
        self.outputs.finish(state.request.envelope.request_id)
        del self.states[state.request.envelope.request_id]
```

The real hot path may move sampling, stopping, and state updates onto the GPU
or into lower-level code. These interface boundaries are still useful for
reasoning about correctness.

## 17. Latency per user

Three user-facing measures answer different questions:

### Time to first token

```text
TTFT =
    request upload/network
  + gateway/auth/rate-limit work
  + render/tokenize/preprocess
  + admission and queue delay
  + scheduler gaps before/during prefill
  + uncached prefill execution
  + first sampling/output processing
  + first response-byte network delay
```

Long prompts primarily increase prefill work. Prefix hits reduce the uncached
part. Queueing can dominate both.

### Inter-token latency or time per output token

For output step `i`:

```text
TPOT_i =
    wait until selected for next iteration
  + batch assembly and host/device overhead
  + decode execution and distributed communication
  + sampling/constraints/stop processing
  + output queue and network effects
```

Decode time can increase with:

- Larger effective batch or more simultaneously running sequences.
- Longer contexts, because attention reads more KV state.
- KV-cache placement, quantization, or offload.
- Distributed communication.
- Structured decoding or expensive logit processors.
- Scheduler fairness and priority.
- Interference from large prefills.

### End-to-end latency

```text
E2E latency
  ~= TTFT
     + sum(inter-token intervals)
     + final flush/close overhead
```

Output length is often the strongest per-request contributor because decode is
serial along the token dimension.

### Why two users see different latency

Even on the same model and replica, users can differ because of:

- Prompt length and prefix-cache hit rate.
- Requested and actual output length.
- Arrival time relative to bursts.
- Priority, quota, and fairness policy.
- Other requests' prompt/decode mix.
- Context length of neighboring decode sequences.
- Adapter or model variant.
- Sampling and structured-output settings.
- Replica placement and network path.
- Client read speed and cancellation behavior.
- Preemption or worker failure.

The right question is not only, "How fast is one forward pass?" It is, "How
often does this request receive a forward pass, in what batches, with what
context length and resource contention?"

## 18. Fairness and per-user controls

Request-count fairness is insufficient. One request can ask for a 100-token
prompt and 20 output tokens; another can ask for a 100,000-token prompt and
10,000 output tokens.

A useful policy can account separately for:

- Prompt tokens admitted.
- Output tokens generated.
- Concurrent live sequences.
- KV bytes held over time.
- Accelerator execution time or estimated work.
- Queue age and deadlines.

Conceptually:

```python
tenant_virtual_time[tenant] += (
    PREFILL_WEIGHT * scheduled_prompt_tokens
    + DECODE_WEIGHT * scheduled_decode_tokens
    + KV_HOLD_WEIGHT * kv_block_seconds
) / tenant.service_share
```

The scheduler can choose eligible tenants with the smallest virtual time while
still protecting decode latency and honoring strict priorities.

Routing affinity can improve prefix reuse for multi-turn conversations, but it
should be a hint, not a correctness requirement. A failed or overloaded
replica must be replaceable.

## 19. Cancellation, retries, and failures

Cancellation is a state transition:

```text
client disconnect
  -> API layer marks cancellation
  -> scheduler stops selecting the sequence
  -> in-flight iteration finishes or ignores its result
  -> output channel closes
  -> KV references and request metadata are released
```

Failure handling depends on whether output has been observed:

- Before the first token, a gateway may retry on another replica if the
  operation and billing semantics are idempotent.
- After partial streaming, transparent retry is difficult because it can
  duplicate or diverge from already emitted output.
- With deterministic replay, the service may reconstruct the prefix, but it
  still needs an explicit stream-resumption protocol and output cursor.
- A distributed worker failure often invalidates the whole replica's current
  batches.

Every request needs a single idempotent retirement path. Success,
cancellation, timeout, rejection-after-reservation, and executor failure must
all eventually release the same owned resources.

## 20. Optional architectural variants

### Disaggregated prefill and decode

```text
request
  -> prefill router
  -> prefill worker pool
  -> KV transfer/storage
  -> decode worker pool
  -> stream
```

This lets prefill and decode use different hardware or scaling policies. It
also introduces KV-transfer latency, placement constraints, remote-cache
ownership, and new failure modes.

### Speculative decoding

A draft mechanism proposes several tokens; the target model verifies them.
The scheduler now budgets proposed and verified tokens, and the KV manager must
commit accepted state while discarding or rolling back rejected state.

### Adapter multiplexing

Requests share base weights but select adapters. The scheduler may group work
by loaded adapter, account for adapter memory, and decide whether adapter load
latency is acceptable.

### Multimodal serving

An encoder or preprocessing service produces image/audio/video features.
Those features add preparation latency, device memory, caching questions, and
possibly a separate scheduler before language-model prefill.

## 21. Observability by stage

| Stage | State/resource | Primary measures |
|---|---|---|
| Gateway | connections, bytes, auth calls | request rate, rejection rate, event-loop lag |
| Preparation | CPU workers, token arrays | tokenize latency, prompt tokens, preprocessing queue |
| Admission | quotas, bounded queue | admitted/rejected, reason, predicted wait |
| Scheduler | waiting/running sequences | queue delay, scheduling delay, batch composition |
| KV manager | blocks and references | free/used/cached blocks, hit rate, evictions, preemptions |
| Prefill | prompt chunks | TTFT, prompt tokens/s, chunk sizes |
| Decode | live contexts | TPOT/ITL, output tokens/s, active sequences |
| Sampling | logits and constraint state | sampling latency, constraint stalls |
| Streaming | socket/send buffers | time to flush, buffered bytes, disconnects |
| Retirement | leases and metadata | leaked requests, cleanup latency, finish reasons |

Histograms should be split by prompt-length bucket, output-length bucket,
model, adapter, priority, cache-hit status, and finish reason. Fleet-wide
averages hide the exact interference effects the scheduler creates.

## 22. A worked trace

Assume one request has a 4,000-token prompt and asks for at most 500 output
tokens.

1. The API server allocates connection and request metadata.
2. Authentication and tenant limits pass.
3. Prompt rendering and tokenization produce 4,000 token IDs.
4. Admission estimates prefill work and up to 4,500 tokens of KV occupancy.
5. The router chooses replica B because it has compatible weights, a short
   prompt queue, and possible prefix locality.
6. The KV manager finds 3,072 reusable prefix tokens and creates a lease
   referencing those blocks.
7. The request enters the prefill queue with 928 unmatched prompt tokens.
8. The scheduler first serves existing decode sequences, then schedules a
   512-token prefill chunk.
9. A later iteration schedules the remaining 416 prompt tokens.
10. Prefill completes; logits for the first output token are sampled.
11. The first token is detokenized and streamed. TTFT ends here.
12. On each later engine iteration, the request competes for one decode slot.
13. Each selected decode step consumes the previous sampled token, appends KV,
    produces next-token logits, samples, checks stopping, and emits output.
14. At output token 173, a stop token is sampled.
15. The engine marks the request finished, flushes the final event, publishes
    eligible prefix blocks, releases the lease, and removes request metadata.
16. The application stores the assistant message in the conversation record.

If the user returns tomorrow, the application constructs a new request from
that stored conversation. Replica B's old KV blocks might still exist, might
have moved to another cache tier, or might be gone. Correctness is unchanged;
only latency differs.

## 23. Questions to ask of any serving design

### State and ownership

- What is the source of truth for conversations?
- Which request states survive process failure?
- What uniquely identifies model-equivalent KV state?
- Is prefix reuse isolated across tenants where required?
- Is cleanup idempotent for every terminal path?

### Admission and capacity

- Which queues are bounded, and by what units?
- Is admission based on requests, tokens, KV bytes, or predicted work?
- What happens when estimated deadlines are infeasible?
- Which overloads return `429`, and which return `503`?
- Can one tenant consume all queue slots or live KV blocks?

### Scheduling

- Is batching static or continuous?
- Are decode requests protected from long prefills?
- How are prompt chunks sized?
- What is the fairness unit?
- What triggers preemption, and how is state recovered?

### Memory

- How much HBM remains after weights and workspaces?
- What is KV bytes per token locally and across the replica?
- Are blocks allocated eagerly or incrementally?
- Are completed prefixes cached, and what evicts them?
- Can KV be offloaded or transferred, and at what latency?

### Latency and reliability

- Can TTFT be decomposed into preparation, queue, and prefill?
- Can TPOT be decomposed into scheduler wait and execution?
- What happens on client backpressure or disconnect?
- Is retry safe before and after partial streaming?
- Which failures invalidate one request, one batch, or the entire replica?

## 24. Compact mental model

When analyzing any inference question, walk through these five ledgers:

1. **State ledger:** What state exists, who owns it, and how long does it live?
2. **Queue ledger:** Where can work wait, what bounds the queue, and who goes
   next?
3. **Memory ledger:** What is static, what grows per token, and what is
   reclaimable?
4. **Compute ledger:** Is this prefill work, decode work, sampling work, or
   communication?
5. **Time ledger:** Which components contribute to TTFT, TPOT, and total
   completion time?

Then trace the request twice:

```text
slow trace: API request -> preparation -> queue -> prefill -> decode -> retire
fast trace: scheduler plan -> KV allocation -> batch -> execute -> update
```

If both traces are clear, most design and debugging questions become local:
you can identify the owner, resource, queue, transition, and metric involved.

## Appendix.a More notes on KV Cache Organization in a Model+Engine Replica

### The Core Concept: GPU-First Architecture

In high-performance inference, **GPU VRAM is the primary, high-bandwidth heap for the KV cache.**

Because modern LLMs are overwhelmingly memory-bandwidth bound during the decode (autoregressive) phase, every single token generation requires fetching the existing KV cache from memory. The bidirectional bandwidth of even the fastest PCIe Gen5 or specialized CPU-to-GPU interconnects is an absolute bottleneck compared to on-device HBM (High Bandwidth Memory).

If an engine had to constantly page blocks in from CPU RAM during a normal decode iteration, token generation speeds would plummet by an order of magnitude. Therefore, the architecture behaves under these rules:

* **The Bound is GPU HBM:** The maximum amount of *actively executing* KV cache is strictly bounded by the free HBM available across the replica's GPUs after the model weights are loaded.
* **CPU RAM is an Overflow Valve (or Optimizer):** CPU RAM is not the primary storage for active sequences. Instead, it acts as a **swap space** for preempted requests or a **warm tier** for an evictable prefix cache.

---

### Sizing the Dynamic Heap on a GPU Replica

To understand exactly how much KV cache capacity a replica has, you can calculate it using a simple structural breakdown.

#### 1. The Static vs. Dynamic Split

When a model replica initializes across a group of GPUs (whether utilizing Tensor Parallelism, Pipeline Parallelism, or both), the total HBM is divided:

$$\text{Total HBM} = \text{Model Weights} + \text{Static Workspaces (CUDA Graphs, etc.)} + \text{The KV Cache Pool}$$

The engine pre-allocates virtually all remaining HBM into a contiguous **Paged KV Cache Pool**, dividing it into fixed-size physical blocks (typically 16 or 32 tokens per block).

#### 2. The Multi-GPU Scaling Math

The total KV cache capacity scales with the number of partitioned heads across your distributed fabric. For a standard transformer model, the exact footprint of **one token's KV cache across the entire replica** is calculated as:

$$\text{Bytes per Token} = 2 \times \text{Number of Layers} \times \text{Number of KV Heads} \times \text{Head Dimension} \times \text{Bytes per Element}$$

> **Note on Architectures:** Modern optimization techniques directly shrink this per-token footprint. Grouped-Query Attention (GQA) or Multi-Query Attention (MQA) drastically reduce the `Number of KV Heads`, while FP8 or FP4 quantization reduces the `Bytes per Element`.

If your replica consists of 8 GPUs mapped via Tensor Parallelism, the model weights are split across them, but the total *effective* KV cache capacity is the sum of the KV block pools across all 8 devices.

---

### How CPU RAM Actually Fits In

So, if the max active footprint is bounded by GPU HBM, what happens when you run out of blocks, and where does that terabyte-scale CPU RAM come into play?

In production engines, CPU memory serves two primary roles:

##### Preemption via Swapping

When the arrival rate of prompts spikes (surpassing what the GPU HBM can hold), the engine's scheduler faces an allocation crisis. To avoid crashing or dropping requests, it **preempts** lower-priority running sequences.
Instead of discarding their computed KV history entirely (which would require an expensive recomputation step later), the engine asynchronously pages those logical blocks out from **GPU HBM $\rightarrow$ CPU RAM**. When slot capacity frees up on the GPU, they are paged back in.

##### Warm Prefix Caching

If your application features long, highly repeatable system prompts, multi-turn chat transcripts, or massive context documents, the engine will hash these token sequences.

* **Active Cache:** Live blocks sit in GPU HBM.
* **Idle/Shared Cache:** When a request finishes, its blocks aren't immediately zeroed. They are held in a virtual cache pool. If GPU memory pressure mounts, the engine evicts these unreferenced blocks to CPU RAM, maintaining a much larger "warm" dictionary of document contexts.

---

### Summary Mental Model

| Attribute | GPU HBM Pool (The Hot Heap) | CPU RAM Pool (The Swap/Cache Tier) |
| --- | --- | --- |
| **Role** | Primary execution memory for all **actively scheduled** iterations. | Overflow valve for **preempted** tasks and a long-term **prefix optimization** store. |
| **Performance Impact** | Runs at terabytes-per-second memory bandwidth; essential for low token latency. | Limited by PCIe/interconnect speeds; used to prevent catastrophic capacity failures. |
| **Hard Limit** | Bounces requests or triggers preemption if this fills up with *active* sequences. | Can scale to terabytes to hold thousands of dormant, cached, or paused user contexts. |

## References to concrete engines

- [vLLM architecture overview](https://docs.vllm.ai/en/latest/design/arch_overview/):
  separates API servers, an engine core responsible for scheduling and KV
  management, and GPU workers that execute model forwards.
- [vLLM optimization and tuning](https://docs.vllm.ai/en/latest/configuration/optimization/):
  discusses KV-pressure preemption and chunked prefill scheduling.
- [vLLM automatic prefix caching](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/):
  explains that prefix reuse accelerates prefill but not generation of new
  tokens.
- [TensorRT-LLM GPT attention and KV cache](https://nvidia.github.io/TensorRT-LLM/advanced/gpt-attention.html):
  describes contiguous and paged KV caches and block management.
- [TensorRT-LLM KV cache reuse](https://nvidia.github.io/TensorRT-LLM/advanced/kv-cache-reuse.html):
  describes reuse of previously computed KV blocks.
- [SGLang paper](https://arxiv.org/abs/2312.07104):
  presents RadixAttention and prefix reuse for structured language-model
  programs.
- [PagedAttention paper](https://arxiv.org/abs/2309.06180):
  motivates block-based KV memory management for serving.
