# MemRL: device-resident retrieval PPO on Frostbite

MemRL compares `none`, `random`, and `learned` retrieval in a JAX/Flax PPO agent. Training and evaluation use
EnvPool's Frostbite XLA backend exclusively. The rewrite preserves the experimental invariants while intentionally
not preserving exact Gymnasium trajectories or old checkpoint compatibility.

Do not start the 10M-step study until both the correctness and performance gates pass. The performance thresholds
at full memory are median paired ratios `random / none <= 2.0` and `learned / none <= 3.0`.

## Setup

Python is pinned to 3.11. EnvPool 1.2.5 and Orbax 0.12.4 are direct locked dependencies.

```bash
uv sync --extra dev                 # CPU validation
uv sync --extra dev --extra cuda12  # RTX/Colab validation
uv run python -c "import envpool, jax; print(envpool.__version__, jax.__version__, jax.devices())"
uv run pytest
uv run ruff check src tests
```

The shared environment factory maps `FrostbiteNoFrameskip-v4`, `ALE/Frostbite-v5`, and related aliases to
`Frostbite-v5`. It explicitly fixes eight per-environment seeds, four-frame skip and stack, 30 reset no-ops, FIRE
reset, episodic life, reward clipping, 84x84 grayscale area resize, and zero sticky-action probability. Raw game
rewards and real game ends come from EnvPool info; PPO boundaries use episodic-life termination.

## Architecture

Retrieval modes carry a 100,000 x D JAX FIFO with embedding, episode, timestep, insertion, size, and write-index
state. `none` does not allocate this state. Each environment samples K=64 candidates before choosing its action. After the environment step, valid
transition tuples are inserted in environment-slot order. Terminal transitions are included; EnvPool auto-reset
calls that discard the supplied action are skipped. Source observation frames and episode/timestep metadata
stay aligned with their transitions. Warm-up sampling is uniform with replacement; once size reaches K, a vectorized fixed-work
Floyd sampler draws a uniform subset without replacement. It never creates a memory-sized permutation or score
vector.

The Atari encoder is Conv -> GELU -> Conv -> GELU -> Conv -> GELU -> Dense(512) -> GELU.
All GELUs use the tanh approximation (`approximate=True`). Encoder layers and both query dense
layers use unit-gain orthogonal initialization with zero biases; actor and critic output gains
remain 0.01 and 1, respectively.

Memory stores `[f(o_t), onehot(a_t), symlog(r_raw[t+1])]`, where `f` is the raw 512D encoder
feature and `symlog(x) = sign(x) * log(1 + abs(x))`. The reward comes from EnvPool `info["reward"]`;
PPO still uses clipped rewards. The width is `D = 512 + action_dim + 1`, resolved automatically from the
environment; an explicit `memory_dim` must match. No component scales or deterministic projection are added.
The observation-only query MLP applies Dense(512) -> GELU -> Dense(D). Full tuples participate in cosine
scoring and context aggregation. Actor and critic take the 512D online observation concatenated with D-dimensional
context; `none` keeps 512D inputs. The float32 FIFO occupies `100000 * D * 4` bytes at full capacity,
plus candidate buffers. A transition's action and outcome cannot affect its own action selection.

The encoder is fixed throughout each rollout and optimized during PPO. The next rollout uses the updated online
encoder; a separate write-only encoder copied once after all PPO optimizer steps would be equivalent and is not
allocated. Existing entries remain detached and unchanged across iterations, so encoder-version drift remains.

Historical embeddings are detached. Candidate tensors are snapshotted in rollout storage and reused unchanged for
all four PPO epochs. Random retrieval is a direct candidate mean: its query parameters exist for architectural
comparability, but ordinary inference executes neither the query MLP nor cosine scoring. Learned retrieval uses the
query and cosine-softmax weighting.

The full 128-step policy/EnvPool/memory rollout is one JIT call. GAE and all four epochs x four minibatches are a
separate JIT call. The host observes one bundled result after rollout and one after PPO. Retrieval rollouts transfer
their 1,024 grayscale frames in one host batch and update a NumPy frame ring using device-recorded physical slots.

## Correctness gate

```bash
JAX_PLATFORMS=cpu uv run pytest
uv run memrl-smoke --wandb-mode disabled
```

The unit suite covers ring insertion and wraparound, warm-up replacement, post-warm-up uniqueness and uniformity,
deterministic PRNG replay, independent environment samples, capacity-independent sampler structure, no self
retrieval, immutable snapshots, detached candidates, query and retrieval-score gradients, random fast-path
independence, exact EnvPool configuration/accounting, diagnostics schemas, checkpoint round trips, analysis, and
matrix gates. `memrl-smoke` runs real EnvPool in all modes and performs two compiled learned iterations.

A release gate must additionally be run on the target RTX 2060: all-mode real smokes, trace/JAXPR inspection for no
host callbacks in rollout or PPO loops, frame-full and frame-less recovery, and continuation from the latter.

## Checkpoints and resume

Orbax writes coherent checkpoint directories. Training state includes parameters, optimizer, device memory, every
PRNG stream, counters, resolved configuration, dependency/Git provenance, W&B identity, and resume history.
Diagnostic frames, per-slot insertion IDs, and validity are optional. Missing frames never invalidate learning
state; `diagnostics/frame_coverage` reports occupied-slot coverage and memory tables mark unavailable images.

Frames are checkpointed by default. Periodic checkpoints are written every 1,000 rollouts (1,024,000 environment
steps), the newest two are retained, and `final/` is preserved. Async writes finish before rotation and process exit.

```bash
uv run memrl-train --retrieval-mode learned --seed 1
uv run memrl-train --retrieval-mode learned --seed 1 \
  --resume-from checkpoints/RUN/step_1024000
uv run memrl-train --retrieval-mode learned --seed 901 \
  --no-save-memory-frames
```

Resume restores learning state but creates a fresh EnvPool handle. It assigns fresh episode IDs, resets active
episode statistics, and records the unavoidable emulator-reset discontinuity. Legacy `.msgpack`/`.npz` checkpoints
are left untouched and rejected; there is no migration path. Schema 3 retrieval checkpoints require layout
`transition_obs_action_symlog_v1`, a positive action count, and the matching resolved tuple width.
Observation-only retrieval checkpoints (including schema 2, 256D, and 512D) are rejected before tensor restoration.
Schema 2 `none` baseline checkpoints remain compatible. Start fresh tuple retrieval runs in new output directories.

Tuple memory requires fresh correctness and performance validation. Its hypothesis is that historical actions and
outcomes provide useful context; implementation checks and timing alone cannot establish learning quality.

Evaluation accepts the one directory and freezes retrieval K/temperature from its training metadata:

```bash
uv run memrl-eval --checkpoint checkpoints/RUN/final --episodes 10
```

Random evaluation samples uniformly. Learned evaluation scores the whole resident memory, samples K from the
resulting distribution, and uses their unweighted mean. Evaluation uses the fixed saved tuple memory without insertion.

## Performance gate

The benchmark launcher runs nine fresh 200k-step processes exclusively, seeds 901-903, in counterbalanced mode
order. It disables W&B and checkpoints while retaining local metrics, normal diagnostics, and host-frame upkeep.

```bash
uv run memrl-benchmark --run-gate --output benchmark-report.json
```

Compilation and the first 100k steps are excluded from the steady-state median. The report pairs each retrieval mode
with `none` by seed, computes the median of three ratios, rejects missing/non-finite/OOM runs, and separately records
cold compilation time plus peak device and host memory. A failed gate requires a full-memory JAX profile followed by
correctness and performance reruns; it blocks all long jobs.

For initial tuple validation, run three fresh sequential 200k-step processes at seed 901 (one per mode) using
these benchmark settings. Report ratios against the same 2x/3x targets, compilation time, available peak memory,
and finite/OOM status. These are preliminary results: the complete seeds 901-903 release gate remains required.
No learning campaign or hyperparameter tuning is part of the tuple implementation validation.

## Concurrency and learning study

Campaign concurrency defaults to one. `--max-parallel 2` requires `--resource-report` from two concurrent learned
canaries demonstrating full host frames, a frame-complete checkpoint, combined peak VRAM at most 4.8 GiB,
`MemAvailable` at least 0.4 GiB, and swap growth at most 256 MiB. Timing matrices are always exclusive.

```bash
uv run memrl-canary --report canary-resources.json
uv run memrl-matrix --max-parallel 1
uv run memrl-matrix --max-parallel 2 --resource-report canary-resources.json
```

`memrl-canary` runs two learned 200k-step processes concurrently, samples GPU/host/swap usage, verifies their
frame-complete final checkpoints, and prints and writes the resource report.

After every gate passes, run seeds 1-3 for all three modes at the frozen 10M-step budget. Every completed episode is
written to `episodes.jsonl` with mode, seed, environment slot, completion step, raw return, and length.

```bash
uv run memrl-analyze \
  --inputs runs/RUN_NONE_SEED1 runs/RUN_RANDOM_SEED1 runs/RUN_LEARNED_SEED1 \
  --output analysis.json
```

Analysis builds the 101-point 0..10M grid from the latest up to 100 completed episodes, backfills leading empty grid
points from the first populated point, and reports normalized trapezoidal AUC, final-window return, raw seed curves,
and all three paired contrasts: learned-random, random-none, and learned-none. With three seeds the output is strictly
descriptive: no p-values, superiority/equivalence claims, null-effect claims, or directional success label.

## Metric fidelity

Each run writes `metric_metadata.json`. Learned similarities/weights, entropy, effective count, temporal fractions,
age summaries, and embedding/context norms are exact rollout reductions. Random query/similarity checks, policy
interventions, representation checks, and final-batch encoder drift are periodic probes. Encoder drift compares
only observation coordinates. Observation/action/reward block norms and selected transition actions/rewards
make component dominance inspectable. Age counts use the fixed
edges `0, 10, 25, 50, 100, 250, 500, 1k, 2.5k, 5k, 10k, 25k, 50k, 75k, 100k`; top-memory tables use final-step
candidates. `retrieval/recent_under_500_fraction` remains an exact scalar.
