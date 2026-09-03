# MemRL: retrieval-augmented PPO on Atari Frostbite

A JAX/Flax implementation of the retrieval-memory experiment specified for
Atari Frostbite. PPO structure, preprocessing, CNN and defaults follow
CleanRL's `ppo_atari_envpool_xla_jax.py`; the pinned upstream source is in
`vendor/` for comparison. The training environment uses Gymnasium so memory
records and diagnostic frames can be maintained explicitly.

## Setup

Python and dependencies are managed entirely by [uv](https://docs.astral.sh/uv/):

```bash
uv sync --extra dev                 # CPU development
uv sync --extra dev --extra cuda12  # NVIDIA CUDA 12 (for Colab)
uv run pytest
```

ALE's current wheels include Atari ROMs. Confirm the environment before a long
run with `uv run python -c "import gymnasium as gym, ale_py; gym.register_envs(ale_py); print(gym.make('ALE/Frostbite-v5'))"`.

## Commands

The baseline has no retrieval parameters in its policy. Random and learned
memory use the same 768-wide actor/critic input and differ only in aggregation.
W&B is online by default; use `--wandb-mode disabled` for local tests.

```bash
# Mandatory all-mode smoke test (small debug budget)
uv run memrl-smoke --wandb-mode disabled

# Three seeds per condition
for mode in none random learned; do
  for seed in 1 2 3; do
    uv run memrl-train --retrieval-mode "$mode" --seed "$seed" \
      --wandb-project memrl-frostbite
  done
done

# Exact whole-memory retrieval evaluation from a checkpoint
uv run memrl-eval --checkpoint checkpoints/RUN/final.msgpack \
  --memory-checkpoint checkpoints/RUN/memory.npz
```

Training writes the resolved configuration, JSONL metrics, parameter
checkpoints, memory state and retrieval frame diagnostics under `runs/` and
`checkpoints/`. W&B receives episodic return/length, SPS, PPO losses, explained
variance and retrieval diagnostics at the environment-step axis.

The diagnostics are grouped so failures can be localized:

- PPO health: return, both losses, policy entropy, old/forward KL, clip
  fraction, explained variance, raw gradient norm and SPS.
- Retriever learning: entropy/effective count, similarity statistics, query
  gradient norm, gradient with respect to scaled retrieval scores, and query
  parameter change per optimizer step.
- Policy use of memory: embedding norm ratio, policy KL and value change versus
  zero memory, plus policy KL versus random and shuffled memory contexts.
- Representation quality: random-pair cosine, dimension variance, pair distance,
  near-duplicate rate, pre-normalization norms and stored/current encoder drift.
- Temporal behavior: global retrieval age histogram, recent-memory fraction,
  same-episode/previous-episode fractions and same-episode timestep distance.

The five primary W&B plots are `charts/episodic_return`,
`retrieval/entropy`, `gradients/query_network_norm`,
`interventions/policy_kl_memory_vs_zero`, and
`drift/stored_current_cosine_mean`.

## Method details

Each current CNN feature `z` is projected without trainable parameters to the
256-dimensional historical embedding `h`; NumPy copies enter a 100,000-step
FIFO and cannot receive gradients. Rollouts store candidate embeddings, IDs,
timesteps and indices. PPO reuses those exact candidates for every update epoch,
so the old and new action probabilities have identical retrieval context.
Learned mode uses a 512→256→256 query MLP and cosine-softmax weights at
temperature 0.1. Random mode takes the candidate mean.

During training candidates are sampled uniformly. Evaluation scores all stored
embeddings, samples K indices from the resulting cosine-softmax distribution,
and takes their unweighted mean as specified. The large raw frame ring is
optional at checkpoint time; diagnostic top matches are emitted as compressed
NPZ batches and W&B tables/images.

`FrostbiteNoFrameskip-v4` is accepted as the experiment-facing name and mapped
to the maintained `ALE/Frostbite-v5` registration when the legacy ID is absent.
Both correspond to no built-in action repeat; the CleanRL max-skip wrapper
performs the four-frame skip.

## Colab

Open `colab_memrl.ipynb`, select a GPU runtime, configure the repository path,
and run the setup cell. It installs uv, syncs the CUDA extra, verifies JAX sees a
GPU, logs into W&B, runs tests and exposes the three-condition launch cell.
For persistent artifacts, set `--output-dir` and `--checkpoint-dir` to mounted
Google Drive paths.

## Local RTX 2060 guidance

A 6 GB RTX 2060 should fit the default 8×128 rollout: the candidate snapshot is
64 MiB, while the FIFO uses about 98 MiB for embeddings and 674 MiB for one
diagnostic grayscale frame per entry in host RAM. JAX preallocation is disabled
and its memory fraction defaults to 0.55. Run a 100k-step timing job before a
full matrix and use the stabilized `charts/projected_hours_per_10m_steps`:

```bash
uv sync --extra cuda12 --extra dev
uv run python -c "import jax; print(jax.devices())"
uv run memrl-train --retrieval-mode learned --seed 901 \
  --total-timesteps 100000 --wandb-project memrl-frostbite-validation
```

Do not infer throughput from the four-step smoke test because JAX compilation
dominates it. Run the nine full jobs sequentially on this card.

## Reproducibility boundary

A successful smoke test verifies execution, gradients, mode switching and finite
metrics. It is not a baseline reproduction or a Frostbite learning result. The
requested experiment requires three complete seeds for each mode with the same
budget, followed by curve aggregation and retrieval-frame review in W&B.
