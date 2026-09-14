<p align="center">
  <img src="assets/johnny-silverfly.jpg" width="440"
       alt="Johnny Silverfly: a cyberpunk fly in a leather jacket on a neon rooftop">
</p>

# Johnny Silverfly

*Son of Stonkfly.*

A fly-connectome simulation that can operate a crypto trading account. Actual neural output, actual Coinbase integration. Profitable learning has not been demonstrated. Every ticker on that billboard is invented; what the flies actually returned is [below](#what-the-measurements-say).

**How it works:** Descriptors of public Coinbase prices — trend, volatility, position in range, acceleration — plus an efference copy of the fly's own last filled trade, are placed on 53 glomerular channels across 2,635 olfactory receptor neurons in the retained **MaleCNS v1.0 graph: 166,700 neurons, 25.6 million connections**. They reach 3,829 of the 4,064 Kenyon cells in two synapses. Account equity reaches 23 LB3c sugar gustatory cells. A fixed neural readout proposes buy, sell or hold, and a custom **Coinbase AgentKit ActionProvider** checks limits and places spot orders through Coinbase Advanced.

The prices are also drawn as a chart and fed to 3,335 brightness and 811 R8 colour receptors, which is how this started. **That pathway has no operating point.** At the shipped inhibitory ratio the chart reaches the mushroom body with two cells; across the whole range of ratios it drives either a third of the Kenyon population or exactly two of it, never anything between. What looked like visual input was saturation, and the olfactory channel exists because of that measurement.

Positive portfolio P&L stimulates 15 identified PAM11 dopamine cells; negative P&L stimulates two PPL101 aversive dopamine cells. A candidate memory rule changes existing KC-to-MBON connections; with no reinforcement it changes none. These are engineered reinforcement signals, **not modeled pain receptors**. Synaptic changes do not establish that it learns to trade profitably. [Model and evidence](docs/model.md).

## What the measurements say

Every number here comes from a read-only probe in `tools/diagnose.py`, including the ones that cancelled planned work. Full write-ups in [validation](docs/validation.md).

- **The network was saturated, and one ratio explained it.** Excitation and inhibition were both set from contact count, so 94% of the layer one synapse from the olfactory receptors fired for every market state. Scaling inhibitory weights by **1.9** holds Kenyon activity at 1.6–2.4%, the range a real mushroom body works in. It was selected on sparseness by a rule declared before the sweep, never on returns.
- **A rally and a crash reached the retina as 98.5% the same image.** Ten of twenty-eight market-state pairs produced a byte-identical Kenyon code. Drawing the price history as a filled area removed all ten collisions, and the fill still ships — but this was the first sign that the visual pathway was saturated rather than informative, which the ratio sweep later confirmed from the other side.
- **Reinforcement responds to timing, not just dose.** Over 60 observations no plastic edge moves without external reinforcement, and scrambling the order of the same reward labels changes the resulting memory more than removing them does.
- **The readout is noise-limited, and widening it does not fix that.** Silencing a random 1% of the network flips the decision four times in five. Three candidate readouts, up to 2,119 cells, all score signal-to-noise below 1 — the market moves them less than silencing unrelated neurons does. So **no decoder change ships**; the bottleneck is upstream.
- **Nothing has traded at a profit.** Not the fixture runs, not the first evolution, whose pre-declared kill criterion rejected its own champion. That result is published rather than retried until it passes.

## Evolving it on a GPU

An evolution is thousands of 500 ms observations, and 95.2% of each one is the spiking kernel. `stonkfly/neural/kernel.cu` is a CUDA port of that kernel which runs one fly per CUDA block, and `--device gpu` puts a whole wave of flies through one launch.

The port is **bit-identical**, not approximate. `tools/gpu_port_check.py` advances the same brain on both kernels from the same state and compares all eighteen mutable arrays element for element after a full 5,000-tick observation; `tools/herd_check.py` does the same claim end to end, comparing profit, every proposal count, fills and Kenyon spikes for the same genomes at the same chronological starts. Getting there meant reconstructing the subnormals this hardware flushes to zero, tabulating every exponential the host keeps at double precision, and spelling the voltage update in round-to-nearest intrinsics so the compiler cannot reassociate a four-term sum.

**9.7 fly-observations a second, against 2.08 for the eight-worker pool.** The kernel is still the largest cost at 61.7%, which is the answer to whether moving it was worth doing; `tools/herd_cost.py` prints the rest of the split.

```sh
python -m tools.gpu_port_check --steps 5000 --batch 84   # the kernel alone
python -m tools.herd_check --genomes 3 --starts 2        # the whole fly
python -m tools.evolve --device gpu --population 84
```

## What is in here

| Path | What it is |
| --- | --- |
| `stonkfly/` | the model, the sensory channels and the live worker; the package keeps the old name, and so do `python -m stonkfly` and the `STONKFLY_*` variables |
| `tools/diagnose.py` | read-only probes; never modifies the graph and asserts so |
| `tools/evolve/` | disclosed profit-selected search over declared free parameters; `--device gpu` runs it as herds |
| `stonkfly/neural/kernel.cu` | the CUDA port of the kernel, proven equal to `kernel.cpp` array by array |
| `tools/gpu_port_check.py`, `tools/herd_check.py` | the two proofs of that, one for the kernel and one for the whole fly |
| `tools/fetch_candles.py` | the only thing outside `stonkfly/` that opens a socket |
| `docs/` | [model](docs/model.md), [validation](docs/validation.md), [operations](docs/operations.md) |

## Run it

Python 3.11 or newer, a C++17 compiler, macOS/Linux/Windows. Allow several GB for the dataset and dependencies; 16 GB RAM recommended.

```sh
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
python -m stonkfly prepare
python -m stonkfly run
```

On Windows, install the Visual Studio Build Tools **Desktop development with C++** workload; the kernel is then built with MSVC automatically. Activate with `.venv\Scripts\Activate.ps1` instead of `source`, and the remaining commands are identical.

Default: **paper trades, real public BTC-USDC data, $100 simulated balance**. No key needed. Local logs, sensory images and resumable brain state go in `runs/paper/`. Ctrl-C stops it; the same command resumes.

For real orders, first create a dedicated Coinbase Advanced portfolio with **at most 100 USDC** and a portfolio-scoped **ECDSA API key with View + Trade, no Transfer**. Copy `.env.example` to `.env`, fill it in locally, then run these commands yourself:

```sh
python -m stonkfly run --live --preflight-only
python -m stonkfly run --live
```

Defaults: $10 maximum order including reserved fees, 24 attempts/day, no shorts or leverage. A $20 drawdown stops new orders; **it does not liquidate holdings or cap further losses**. [Operation and recovery](docs/operations.md).

```sh
python -m stonkfly status
python -m pytest -q
```

The repo does not come funded or connected to anyone’s account. Live execution needs your local credentials and explicit opt-in.
