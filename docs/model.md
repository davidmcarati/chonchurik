# What is actually modeled

This is a wiring-constrained spiking-network experiment. The connectome supplies anatomy, not a complete living fly, calibrated physiology, or a trading strategy. No LLM selects trades. No price rule overrides the neural proposal with a different action.

## Anatomy and dynamics

The [MaleCNS v1.0 release](https://male-cns.janelia.org/download/) provides the male brain and ventral nerve cord. Import retains every assigned neuronal superclass, including uncertain classes, while excluding explicit glia and unresolved segmentation objects. It retains all released edges between those entries, including weak edges and self-connections: **166,700 nodes, 25,582,938 directed connections, 124,177,617 synaptic contacts**. “Full retained” describes this inclusion policy; it does not mean every biological synapse was reconstructed.

The importer verifies SHA-256 source files and every compiled graph array against committed locks. Transmitter annotations and cell order are checked too. Neuron IDs remain integers. Source files are downloaded separately under their upstream license.

The native kernel integrates approximate leaky integrate-and-fire cells at **0.1 ms**. It uses 20 ms membrane and 5 ms synaptic time constants, a −45 mV threshold, 1.8 ms transmission delay and 2.2 ms refractory period. Contact count times 0.275 sets initial synaptic magnitude. ACh is assigned excitation; GABA, glutamate and histamine inhibition, with explicit positive fallback for unresolved signs. This is a coarse sign proxy, not receptor-specific physiology.

Inhibitory magnitudes are then scaled by **1.9**. Setting excitation and inhibition from one quantity leaves this network saturated: 94% of the layer one synapse from the olfactory receptors fires for every market state, and 41–47% of Kenyon cells with it. The scaled ratio holds Kenyon activity at 1.6–2.4% across market states, which is the range a real mushroom body works in. Sign, wiring and relative magnitudes within the inhibitory set are untouched; only the ratio moves, and the ratio was always a model choice rather than a measured property of the reconstruction. The value was selected on sparseness and code overlap by a rule declared before the sweep, never on trading returns. See [validation](validation.md). KC rest is −60 mV with an 8 mV adaptation increment decaying over 200 ms; other cells rest at −52 mV.

Pure dopamine, serotonin and octopamine annotations deliver modulatory traces along their retained edges instead of generic fast excitation. Only the specified memory rule consumes selected dopamine activity; most modulatory effects are unmodeled. Cotransmission and receptors remain incomplete. Keeping an edge in the graph does not establish that all its biological effects are reproduced.

The event-driven kernel avoids unnecessary subthreshold updates; it does not prune the graph or enlarge the integration timestep. This accelerates model execution, not biological time.

## What the fly sees

Coinbase public completed one-minute candle closes initialize the price history. Subsequent observation midpoints are appended. A fixed 320×180 chart shows past prices, pair name and current bid/ask; the simulator receives its **RGB pixels**, not raw prices or indicators. The chart is locally rendered, not a capture of a logged-in Coinbase account. It shows prices only; account state reaches the network through the separate channels described below, not through the picture.

The price history is drawn as a **filled area** under the curve, coloured by whether the window closed above or below where it opened. The 3-pixel polyline this replaced put the market on roughly 1.5% of the mapped receptors, and a rally and a crash reached the retina as 98.5% the same image; ten of twenty-eight market-state pairs then produced a byte-identical Kenyon-cell code. Filling the area removed all ten collisions. It is a display adapter chosen on a measured separation score, not on trading returns; see [validation](validation.md).

3,335 mapped R1–R6 cells receive linear-sRGB luminance; 811 mapped R8 cells receive blue/green proxies. Sample locations are inferred from contacts with column-annotated visual cells, using overlapping left/right viewports. Those two viewports do not sample comparable populations: the release contains 1,112 left and 2,265 right R1–R6 cells, so the right eye receives about 2.1x the summed drive, and the left sample points sit much higher in the frame. See [validation](validation.md) for what that does and does not explain. Unmapped receptors get no invented optical input. Photoreceptors and lamina are graded in real flies; using spikes, RGB channels, saturating current and a 12 mV-equivalent lamina bias is an explicit display adapter, not validated retinal physiology.

Existing R8→aMe12 connections use a net excitatory sign motivated by [Xiao et al., 2023](https://doi.org/10.1038/s41586-023-06681-6); transferring that result to these reconstructed cells and contact-count magnitudes remains an assumption. The initial dark chart barely activated KCs in our probe. We changed the chart to a light background, without changing neural currents or fitting to trading returns. Display sensitivity is a major confound to test.

## What the fly smells and tastes

Three further channels drive populations the graph has always retained and nothing previously touched. All three are engineered assignments with no biological content: a real ORN_DA1 answers cVA, and a fly tastes sugar, not a balance. What is borrowed is the architecture, not the chemistry.

**Market descriptors to olfactory receptor neurons.** MaleCNS v1.0 retains 2,635 ORNs in 53 glomerular channels, and they reach 3,829 of the 4,064 Kenyon cells in two synapses through the antennal lobe — the sparse-coding input the mushroom body is built around, which the visual pathway does not provide. Five scale-free descriptors of the price history (fast trend, slow trend, volatility, position in range, acceleration) are placed on glomeruli as a population code, one peak per descriptor. Glomeruli are taken in alphabetical order and split into equal bands, so the assignment is fixed before any price is read. Window lengths, deflection scales, tuning width and peak current are declared model choices, none of them fitted to returns.

**The fly's own last filled trade** occupies a sixth band. It is delivered at the *next* observation, since nothing can be smelled before it happens. HOLD, a vetoed proposal and the first observation of a run share one resting value, so "no trade" is a single odour rather than three.

**Account equity to LB3c sugar gustatory neurons.** The 23 retained LB3c cells receive equity relative to the ledger's accounting anchor, squashed so an untouched account is a definite resting taste rather than an absence of input. An empty account is a real total loss; a missing or unusable reference reads as resting.

This last channel is a **change to what the experiment discloses to the network**. Until it existed, the fly never received portfolio state: the chart excluded balances and P&L by construction, and the only profit-derived input was the momentary reward/aversive dopamine pulse below. The network now receives account equity continuously, as a graded input, at every observation. Every claim about what it learned must be read against that. It remains an *input*: nothing here selects an action, replaces a proposal or introduces a profit term into the decoder.

By default each market observation advances **500 ms of neural time**, regardless of elapsed wall time. Wall observations are at least 60 seconds apart. That is a deliberately compressed market-to-neural clock, not real-time fly physiology. Eligibility and decay operate in neural seconds. Multiple optional assets are presented in a fixed round-robin schedule; the network does not choose which asset is shown.

## How a neural spike becomes an order

Over each observation, mean right DNp20 firing minus mean left DNp20 firing is decoded as follows:

| Neural measurement | Proposal |
| --- | --- |
| Difference ≥ 2 Hz, with at least one DNpe017 spike | Buy |
| Difference ≤ −2 Hz, with at least one DNpe017 spike | Sell |
| Otherwise | Hold |

This is an engineered interface, not a discovery of “buy neurons.” The mapping is fixed and reads only spike counts. Selected cell IDs appear in the local audit log. Persistent one-sided proposals can therefore be an artefact rather than market insight. They were later measured: the two DNp20 cells receive net drive within 1% of each other, so the cause is not a turning bias in the circuit but a 2 Hz threshold resolving a 2-spike margin on a quantity whose noise is larger than the margin. A rally and a crash also reach the retina as 98.5% the same image, so there is little directional content for the threshold to resolve. See [validation](validation.md).

The guard can reject a proposal for price, budget, inventory, timing or account-state reasons. It cannot replace the proposal or manufacture a profitable policy. AgentKit supplies the ActionProvider/Action interface; our custom provider bridges the separate [Coinbase Advanced exchange API](https://docs.cdp.coinbase.com/coinbase-app/advanced-trade-apis/rest-api). Built-in AgentKit on-chain wallet swaps are not used.

## What changes with profit and loss

At the next observation, equity is cash plus holdings marked at the current bid. Its change since the last observation includes booked fees and unrealized price changes. A change of at least +0.01 USDC schedules a **200 ms, 40 mV-equivalent** artificial current into all **15 PAM11 (α1)** cells. A change of at most −0.01 USDC schedules the same pulse into the **two PPL101 (γ1pedc)** cells. The amplitude is the same for both, but the populations are not: at the 20 this replaces, the reward compartment received 14.7× the per-cell drive the aversive one did, purely because it has fifteen cells against two. The pulse is binary above the threshold, not proportional to profit. The deadband is per observation; tiny changes are not accumulated into a later pulse.

This is feedback about portfolio value, not evidence that the latest action caused that change. Holding an asset can produce either signal. Fees count as a loss. Deposits or unexplained balance changes halt execution instead of becoming rewards. Positive P&L need not be realized profit.

The candidate memory rule acts on **7,835 existing KC→MBON07/MBON11 edges**. It adapts a baseline-centered anti-Hebbian rate rule from [Huang, Luo et al., 2024](https://doi.org/10.1038/s41586-024-07819-w): recent KC activity followed by dopamine tends to depress eligible connections; the reverse timing can potentiate them. Actual network spikes supply KC/DAN rates in bins of at most 10 ms. No price or profit value directly edits a synaptic weight.

The 1-second eligibility traces, 1,800-second memory decay, 50 ms efficacy filter, gain 0.001 and efficacy bounds of 0.1–2× baseline are declared model choices. The anatomy-derived DAN-to-MBON contact fractions distribute modulation within each compartment. They are not measured dopamine concentrations or receptor kinetics.

The PAM11/MBON07 compartment is motivated by [Ichinose et al., 2015](https://elifesciences.org/articles/10719). Applying one rule to both α1 and γ1pedc compartments in this male reconstruction is **our unvalidated extension**, not a replication of either paper. Real fly dopamine can have context-dependent effects. “Profit dopamine” and “loss dopamine” are engineered assignments. Pain receptors, subjective pain, pleasure and consciousness are not modeled or measured.

## What would count as learning

The implementation can demonstrate that sensory input reaches memory cells, that selected dopamine cells spike, and that temporal pairing changes eligible synapses. Those are mechanism checks. Even when weights change, useful credit assignment through the fixed trade decoder is unproven.

To claim learned trading behavior requires held-out chronological market replay, independent starts, frozen-weight and shuffled-reinforcement controls, fees/slippage, equal budgets, retention, and loss of benefit after resetting learned weights. Compare to cash and simple exposure baselines as well: rising crypto prices alone can make any buyer look skilled. Avoid selecting a lucky run or tuning on the test period.

**No profitable learning, strategy improvement, biological replication, or live-funded performance has been demonstrated by this repository’s tests.** See [validation](validation.md) for the narrower checks actually performed.
