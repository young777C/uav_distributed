# Nature-Structured Peer Review

## D-EPA-RHP Multi-UAV Redesign: Distributed Collaborative Decision-Making with VoI-Driven Information Synchronization

**Manuscript under review (pre-submission assessment)**

**Review framework**: 3 independent reviewers + 1 Editor's cross-synthesis report
**Evaluation dimensions**: Originality | Scientific Importance | Interdisciplinary Readership | Technical Soundness | Readability for Nonspecialists
**Calibration benchmarks**: 2024–2026 state-of-the-art — a systematic search identified 15 papers across 5 categories. Key benchmarks include: Zhu et al. (2024, *Science Robotics*) — SoNS field-validated swarm; Ma et al. (2024, *Nature Machine Intelligence*) — MDPO scalable MARL; Arul et al. (2024, *IROS*) — selective multi-robot communication; Xu & Tzoumas (2024/2025, *IEEE T-RO*/CDC) — submodular distributed optimization; Zhang et al. (2026, *AAAI*) — VIL2C VoI-aware MARL; Sao et al. (2025, *IEEE ICUAS*) — event-driven CBBA; Chiariotti & Fabris (2025) — VoI-aware formation control; Pan & Luo (2026) — information-theoretic phase transition; Talamali et al. (2021, *Science Robotics*) — "when less is more."

---

## Reviewer 1: Multi-Robot Systems & Distributed Coordination Expert

**Expertise**: Distributed task allocation, multi-robot coordination, auction/consensus algorithms, field robotics
**Familiar venues**: IEEE T-RO, IJRR, ICRA, IROS, JFR, DARS

---

### Overall Assessment

**Recommendation**: CONDITIONAL ACCEPT (Major Revision Required)

This manuscript proposes to extend a single-UAV distributed planning framework (D-EPA-RHP) to multi-UAV teams, with the central hypothesis that VoI-gated information synchronization will exhibit a cross-over point — becoming net-beneficial at some team size N* despite being net-harmful at N=1. The proposed experimental design systematically varies team size (N), communication budget (B), topology (T), and degradation level (C) to characterize the marginal synchronization benefit MSB(N).

The framing is timely and the gap identification is precise. However, the current draft has three structural weaknesses that must be addressed: (1) insufficient engagement with the 2024–2026 state of the art, which has moved significantly beyond CBBA (2009); (2) the single-UAV results (R_task = 0.29 vs Greedy = 0.79) present a 4.6× performance gap that exceeds the theoretical PoA bound of 2× — the manuscript must explain why; (3) the experimental infrastructure for multi-UAV appears not to exist yet, which means the claimed experimental design is a proposal, not an execution.

### Dimension-by-Dimension Evaluation

#### Originality: ★★★★☆ (4/5 — High)

**Strengths**:

The core conceptual contribution — framing "how much communication is enough" as the central research question rather than "how to communicate better" — is genuinely novel within the multi-UAV coordination literature. While the AoI community asks when data becomes stale, and the CBBA community asks how to converge faster, neither asks the marginal-value question: *at what team size does the marginal benefit of an additional synchronization event exceed its marginal communication cost, and how does this threshold shift with degradation level?*

The four-variable experimental design (N × B × T × C) is also novel. To my knowledge, no existing paper simultaneously varies team size, communication budget, topology, and degradation level. The closest work — Xu & Tzoumas (2025, *IEEE T-RO*) — varies neighborhood size and communication cost in submodular optimization but does not study the N-transition. AC-DMTA (2026, *MDPI Drones*) varies communication frequency but only for team sizes implied by the task set. Neither provides the continuous characterization of the centralization-to-distribution spectrum that this work proposes.

**Weaknesses**:

The VoI-gating mechanism itself is not fundamentally new. Becker et al. (2009) and Carlin & Zilberstein (2009) established the decision-theoretic VoI framework for multi-agent communication 15+ years ago. Zhang et al. (2026, *AAAI*) recently applied VoI-aware scheduling to MARL communication latency. The novelty claim must be more precisely scoped: *the specific application of VoI-gating to multi-UAV task allocation under communication degradation, with systematic marginal-benefit characterization across (N, B, T, C)* — not "VoI-driven communication is new."

#### Scientific Importance: ★★★★★ (5/5 — Exceptional)

**Strengths**:

The question "how much communication is enough for multi-UAV teams" addresses an explicitly acknowledged gap in the 2024–2025 survey literature. The *MDPI Drones* January 2025 survey explicitly identifies "provably robust algorithms that gracefully degrade under bandwidth reduction, packet loss, delay, and jamming" as a "significant research gap." The AC-DMTA paper (2026) similarly states that "the consistency certificate thresholds are binary; no continuous analysis of how performance degrades with increasing staleness."

Furthermore, Pan & Luo (2026) proved that multi-agent success probability decays exponentially with the minimum cut cost of the task constraint graph — P(success) = O(2^(-Ω(C_min))) — establishing that structural information bottlenecks are fundamental, not algorithmic. This work could provide the first empirical characterization of where that phase transition occurs for a concrete multi-UAV task allocation problem, bridging the gap between information-theoretic theory and robotic practice.

The military relevance is substantial and well-documented (Ukraine EW causing 30–95% drone attrition; DARPA CODE/OFFSET; DoD Replicator). The civilian relevance to infrastructure-less disaster response (Hurricane Helene 2024) provides a humanitarian framing.

**Weaknesses**:

The manuscript must be more explicit about what a negative result would contribute. If MSB(N) is flat or negative at all N — meaning even multi-UAV cannot rescue distributed coordination — is that still a contribution? The answer from this reviewer is *yes, potentially a more important one*, but the manuscript hedges rather than embracing this possibility. Compare with Talamali et al. (2021, *Science Robotics*), which made its reputation precisely by showing that *less* communication improves swarm adaptability — a counterintuitive negative result. This manuscript should similarly frame both positive and negative outcomes as equally valuable.

#### Interdisciplinary Readership: ★★★★☆ (4/5 — High)

This work sits at the intersection of four active communities: (1) multi-robot coordination (ICRA/IROS/T-RO), (2) information theory / AoI (IEEE JSAC/TCOM), (3) multi-agent reinforcement learning (NeurIPS/ICML/AAAI), and (4) UAV/drone systems (Drones/AIAA/IEEE Access). This is a genuine strength — the marginal-benefit framing provides a Rosetta Stone connecting these communities.

However, the interdisciplinary reach is currently under-exploited. The AoI community (Kaul, Yates, Costa, Sun et al.) has developed rigorous metrics for "how old is your information" but rarely applies them to task allocation decisions. The MARL community has developed "when to communicate" architectures (CommNet, IC3Net, TarMAC, DCC) but lacks formal marginal-value guarantees. The robotics community has CBBA and its derivatives but assumes communication is free. This manuscript could be the bridge paper, but it must explicitly import terminology and metrics from each community.

#### Technical Soundness: ★★★☆☆ (3/5 — Requires Strengthening)

**Critical concerns**:

1. **The single-UAV performance gap (R_task = 0.29 vs Greedy = 0.79, a 4.6× difference) exceeds Thakoor et al.'s (2020) PoA bound of 2×.** If the gap is implementation-dependent (the identified 7 bugs), the manuscript must demonstrate that the post-fix algorithm is within the theoretical bound. If it still exceeds 2×, the theoretical framework does not apply to this algorithm, and the PoA argument must be withdrawn or qualified.

2. **The proposed experimental design requires multi-UAV infrastructure that does not currently exist in the codebase.** This is a feasibility concern, not a conceptual one, but it affects the timeline credibility. The manuscript should include a brief "infrastructure roadmap" acknowledging what needs to be built and what simplifications are planned (e.g., shared observation space in a single-process simulator vs. true distributed processes).

3. **Seed and episode counts are underspecified.** The single-UAV degradation scan used 1 seed × 3 episodes. The proposed multi-UAV design must specify seed × episode counts for each (N, B, T, C) cell, with justification of statistical power. Henderson et al. (2018, *AAAI*) demonstrated that 3–5 trials produce unreliable rankings in RL. For a claim about a "cross-over point" N*, statistical significance testing between adjacent N values is essential.

4. **The MSB(N) metric conflates two effects:** (a) the benefit of adding a UAV (which may be positive simply due to parallelism) and (b) the benefit of synchronization among UAVs (which is what VoI-gating affects). The metric should be decomposed:

```
MSB(N) = [R_task(N, synchronized) − R_task(N, isolated)]
       = [R_task(N, sync) − R_task(N−1, sync)]   ← marginal UAV benefit
       − [R_task(N, isolated) − R_task(N−1, isolated)]  ← baseline
```

Or equivalently, an ablation comparing VoI-gated vs. periodic (fixed-frequency) communication at the same bandwidth budget.

**Minor concerns**:

- The C3 degradation parameters (distance_loss_max, blackhole_extra_loss, jitter_sigma) should be mapped to real-world equivalents (e.g., "C3-Medium corresponds approximately to a 2 km urban BVLOS link with 30% building occlusion"). This aids reproducibility and domain transfer.
- The G2-M2 task case (100 clustered POIs) is a single scenario. Generalization to G1 (uniform) or G3 (mixed) should be discussed, even if not tested.

#### Readability for Nonspecialists: ★★★☆☆ (3/5 — Needs Work)

The marginal-benefit framing ("how much communication is enough") is intuitively accessible and should be foregrounded. However, the current manuscript uses acronym-heavy technical prose (VoI, D-EPA-RHP, C3, G2-M2, MSB(N), PoA) that will lose readers from adjacent fields. A one-page "Conceptual Introduction" with a simple diagram — the hypothetical S-curve of MSB(N) with four regimes (Starvation → Growth → Saturation → Retrograde) labeled with plain-language explanations — would substantially improve accessibility.

The Ukraine EW statistics (30–95% drone loss rates) provide a powerful, concrete entry point. Lead with this rather than burying it.

### Key Questions for Authors

1. If MSB(N) is negative at N=2, do you stop the experiment or continue to larger N? What is the principled stopping criterion?
2. How do you distinguish "VoI-gating is beneficial" from "any selective communication is beneficial"? Should a random-sampling communication baseline be included?
3. The PoA ≤ 2 bound assumes a congestion game structure. Do your task scenarios satisfy this assumption? If not, what alternative theoretical protection exists?

---

## Reviewer 2: Information Theory & Communication Networks Expert

**Expertise**: Age of Information, event-triggered control, networked control systems, information-theoretic limits, wireless communication for autonomous systems
**Familiar venues**: IEEE TAC, Automatica, IEEE JSAC, IEEE TCOM, IEEE TIT, IEEE INFOCOM

---

### Overall Assessment

**Recommendation**: MAJOR REVISION (Fundamentally Promising)

This manuscript attempts something unusual and valuable: bridging the robotics task-allocation literature (which largely ignores communication cost) with the information-theoretic communication literature (which largely ignores task-level semantics). The core insight — that the *semantic value* of a synchronization message depends on whether it changes a UAV's action, not on how many bits it contains — is precisely the argument of "semantic communication" (Weaver 1949, recently revived by Calvo-Fullana & How 2022, Zhang et al. 2026). The VoI-gating mechanism is an operationalization of semantic communication for UAV task allocation: transmit only when the *decision impact* exceeds the *transmission cost*.

What prevents me from recommending acceptance is threefold: (a) the relationship to the AoI literature is almost entirely undeveloped; (b) the proposed experimental design omits a critical control condition (fixed-rate communication at equivalent bandwidth); (c) the VoI threshold's theoretical properties (convergence, stability under noise, minimum inter-event time) are unanalyzed.

### Dimension-by-Dimension Evaluation

#### Originality: ★★★★☆ (4/5 — High)

The VoI-gating mechanism can be understood as a **semantic event-triggered communication scheme**: an event is triggered when |P_GCS − P_UAV| > θ_t, where P is the action-conditional effective completion probability. This is structurally analogous to state-based event-triggered control (Tabuada 2007, *IEEE TAC*) but applied in the *semantic* (task-relevant belief) space rather than the *signal* (state estimation error) space. This semantic-level triggering is novel in the multi-UAV context.

The connection to Witsenhausen's counterexample (1968, *SIAM J. Control*) — which proved that nonlinear control policies are strictly optimal for decentralized stochastic control, and that communication serves a *dual role* (control signal + information signal) — is noted but underdeveloped. The VoI gate is precisely such a nonlinear policy: it transmits (control action) only when the information content of the message justifies the communication cost. Making this theoretical lineage explicit would strengthen the originality claim.

#### Scientific Importance: ★★★★☆ (4/5 — High)

The AoI-minimal-control community has converged on an insight directly relevant here: **minimizing AoI is not the same as maximizing task performance**. The concept of "Age of Incorrect Information" (Maatouk et al. 2020) or "Value of Information" (Alawad & Kaul 2024) recognizes that stale information may still be correct (and hence have positive value), while fresh information may be irrelevant (and hence have zero value). The D-EPA-RHP VoI gate is an implicit AoII/VoI implementation in the UAV task domain.

This connection is both scientifically important and currently unexploited. If the manuscript explicitly frames the VoI gate as a semantic AoII metric for multi-UAV task allocation, it would:
- Import rigorous tools from the AoI community (stochastic hybrid systems models for inter-event time analysis, Lyapunov-based convergence proofs)
- Export a concrete application domain (UAV task allocation) that the AoI community lacks
- Create a template for bridging communication theory and robotics that neither community currently provides

#### Technical Soundness: ★★★☆☆ (3/5 — Critical Gap)

**Critical gap — No analysis of Zeno behavior**:

Scheres et al. (2023, *IEEE TAC*) proved that measurement noise can induce Zeno behavior (infinite triggering in finite time) in event-triggered control schemes, and that a *strictly positive uniform lower bound on inter-event times* is necessary for implementability. The D-EPA-RHP VoI gate — |P_GCS − P_UAV| > θ_t — is vulnerable to the same pathology under C3 jitter. The experiment already shows evidence: n_feedback ≈ 3,900–4,300 per episode (10,001 steps), meaning ~39–43% of steps trigger feedback. This is not "selective communication" — it is effectively continuous communication with a 60% duty cycle.

The manuscript must:
1. Analyze whether the VoI threshold guarantees a minimum inter-event time under the C3 noise model
2. If not, propose a regularization mechanism (e.g., hysteresis threshold, minimum dwell time, or Scheres-style space regularization)
3. Compare the resulting communication rate against fixed-rate periodic communication at equivalent bandwidth

**Critical gap — No fixed-rate baseline**:

The experimental design compares VoI-gated communication against *no communication* (isolated) and *unlimited communication* (full sync). It omits the most informative baseline: **fixed-rate periodic communication at equivalent bandwidth**. Without this baseline, one cannot distinguish "VoI-gating is beneficial" from "any communication at this rate would produce the same result." This is a fundamental confound. The design must include a periodic-communication baseline matched to the VoI-gated average communication rate.

**Critical gap — The VoI threshold θ_t**:

The threshold's dynamics (adaptation via H, θ, T parameters) are governed by an "adaptive schedule" that was previously broken (Bug #3) and later fixed. The manuscript must analyze: (a) what is the effective θ_t distribution over an episode under C3? (b) is θ_t adapting to signal or to noise? (c) would a fixed, tuned θ_t perform equivalently?

If θ_t adaptation provides no benefit over a fixed threshold (after tuning), the "adaptive" claim collapses. This is testable and should be tested.

#### Interdisciplinary Readership: ★★★★★ (5/5 — Exceptional)

This is the strongest dimension of the manuscript. The work naturally spans:
- **Robotics** (task allocation, path planning)
- **Information theory** (semantic communication, VoI, AoI/AoII)
- **Control theory** (event-triggered control, decentralized stochastic control)
- **Military/defense** (contested-environment UAV operations)

This four-way intersection is rare. Most papers sit in one or two of these communities. The marginal-benefit framing — "how much communication is enough?" — is a question that naturally recruits readers from all four. If the manuscript successfully imports the formal tools from AoI/event-triggered control and exports the UAV task allocation application domain, it could become a bridge paper cited across these communities.

#### Readability for Nonspecialists: ★★☆☆☆ (2/5 — Substantial Revision Needed)

The information-theoretic framing (VoI, AoI, semantic communication) is powerful but the manuscript currently buries it in domain-specific UAV jargon. A communication theorist reading this would ask: "What is G2-M2? Why should I care about C3? What is R_task?" Conversely, a roboticist would ask: "What is AoII? Why does Zeno behavior matter?"

The fix is a clear conceptual diagram showing: Information Source (UAV observations) → Semantic Encoder (VoI Gate) → Channel (C1/C2/C3 degradation) → Semantic Decoder (GCS probability update) → Decision (task allocation). This Shannon-Weaver semantic communication pipeline immediately makes the work accessible to both communities and positions the VoI gate as a *semantic source encoder* — a concept both communities understand.

### Key Questions for Authors

1. Under C3 jitter, does the VoI gate guarantee a positive minimum inter-event time? If not, what regularization do you propose?
2. Have you tested whether a fixed periodic communication schedule at the same average rate as VoI-gated communication achieves equivalent R_task?
3. Can the adaptive θ_t be replaced with a fixed, tuned threshold without performance loss?
4. Would a hysteresis threshold (trigger on crossing θ_high, reset on crossing θ_low) reduce the ~39% feedback rate while preserving information value?

---

## Reviewer 3: Machine Learning & Multi-Agent Reinforcement Learning Expert

**Expertise**: MARL, CTDE, emergent communication, graph neural networks for multi-agent systems, learned coordination
**Familiar venues**: NeurIPS, ICML, ICLR, AAAI, AAMAS, CoRL

---

### Overall Assessment

**Recommendation**: REJECT (with encouragement to resubmit after addressing fundamental comparison gaps)

I appreciate the ambition of this work and the systematic experimental design. However, I have a fundamental concern: the manuscript proposes to compare a hand-designed VoI-gating mechanism against baselines, but it does not engage with the past five years of MARL research on *learned* selective communication. The claim that VoI-gating is a principled alternative to "black-box" learned communication is defensible — but only if the comparison is made. Currently, the manuscript ignores an entire literature that asks the same question ("when should agents communicate?") and has developed sophisticated answers.

### Dimension-by-Dimension Evaluation

#### Originality: ★★★☆☆ (3/5 — Moderate)

From the MARL perspective, the VoI-gating mechanism is a **hand-designed, causally-motivated communication policy**. The MARL community has explored this space extensively:

- **RIAL/DIAL** (Foerster et al. 2016, *NeurIPS*): Differentiable inter-agent learning — agents learn *what* to communicate through backpropagation
- **CommNet** (Sukhbaatar et al. 2016, *NeurIPS*): Continuous vector communication with learned averaging
- **IC3Net** (Singh et al. 2019, *ICLR*): Gating mechanism for *when* to communicate, learned via RL
- **TarMAC** (Das et al. 2019, *ICML*): Targeted multi-agent communication with signature-based attention
- **DCC** (Ma et al. 2021, 2025): Decision Causal Communication — agents learn to communicate only with neighbors whose messages *causally change* their action policy. This is the closest learned equivalent to VoI-gating — the causal intervention test (front-door criterion) serves the same function as the VoI threshold but is learned rather than hand-designed.
- **VIL2C** (Zhang et al. 2026, *AAAI*): VoI-aware low-latency communication using KL-divergence of action distributions — perhaps the most direct comparison point, as it explicitly uses a VoI metric within a learned MARL framework.

The manuscript's VoI-gating mechanism is a *model-based* alternative to these *learned* approaches. This is a valid intellectual position — model-based methods have interpretability, sample efficiency, and safety advantages — but the manuscript must (a) acknowledge the existence of these learned alternatives, (b) explain why a hand-designed VoI gate is preferable in this domain, and (c) ideally include at least one learned baseline (e.g., a simple IC3Net-style gating policy trained on the same scenario) as a comparison point.

Without this engagement, the contribution appears to reinventing a 2019-era idea (IC3Net's communication gate) using a 2009-era framework (Becker's VoI) — without acknowledging either.

#### Scientific Importance: ★★★★☆ (4/5 — High)

The question "when should agents communicate?" is central to MARL and remains unsolved. The MARL community's learned communication approaches suffer from well-documented pathologies: (a) positive signaling drift (agents learn to communicate continuously to maximize reward, defeating the purpose); (b) generalization failure (communication policies trained on N agents fail at N+1); (c) computational cost (learning communication policies doubles the training complexity).

A model-based VoI approach that provides interpretable, provably bounded, and generalizable communication decisions addresses real gaps in the MARL learned-communication literature. The VIL2C paper (Zhang et al. 2026) began bridging this gap by incorporating a VoI metric into MARL, but theirs is a hybrid approach (VoI used to allocate learned communication resources) rather than a pure model-based alternative.

If the manuscript can demonstrate that hand-designed VoI-gating matches or exceeds the performance of learned communication policies on the same benchmark — with better generalization, interpretability, and no training cost — this would be a significant contribution to *both* the robotics and MARL communities.

#### Technical Soundness: ★★☆☆☆ (2/5 — Major Gaps)

**Gap 1 — No learned baseline**:

The experimental design compares against Greedy, EPA-Centralized, Event-Dual-RHP, and Periodic-Dual-RHP — all hand-designed algorithms. There is no learned baseline. At minimum, a simple IC3Net-style gating policy (communication gate: transmit if sigmoid(MLP(observation)) > 0.5) trained via PPO on the same scenario should be included. Failing to compare against learned communication is like proposing a new optimizer without comparing against Adam.

**Gap 2 — No communication-incentive analysis**:

In MARL, communication policies are learned, which means the reward function implicitly defines the "value" of communication. The manuscript's VoI gate uses an explicit probability-difference metric. A comparison of *what information each approach decides to transmit* (and when) would be highly revealing: does the learned gate converge to something resembling the VoI metric? If so, the VoI gate is validated as a principled target. If not, one of them is wrong — which one, and why?

**Gap 3 — Scalability claims without CTDE context**:

The manuscript claims scalability advantages of distributed over centralized. But the MARL community already solved this: CTDE (Centralized Training with Decentralized Execution) provides centralized-level policy quality with decentralized execution. Algorithms like QMIX, MAPPO, and HAPPO train with global information but execute with local observations — no communication needed at execution time. The manuscript must explain why hand-designed VoI-gating is preferable to CTDE for this domain. If CTDE-trained policies match centralized performance without any execution-time communication, the entire problem framing collapses.

**Gap 4 — No communication topology learning**:

Recent MARL work (Jiang et al. 2020, "Graph Convolutional RL"; Naderializadeh et al. 2021, "Learning to Share") has shown that *who* to communicate with is as important as *when*. The proposed star topology (UAVs ↔ GCS) is a design choice, not a conclusion. A learned topology (graph attention determining communication edges) might discover that UAV-UAV communication is more valuable than UAV-GCS communication in certain scenarios.

#### Interdisciplinary Readership: ★★★☆☆ (3/5 — Unrealized Potential)

The robotics-MARL intersection is active and growing (CoRL, RA-L + ICRA MARL workshops, AAMAS robotics tracks). However, this manuscript currently reads as a robotics paper with no MARL awareness. Adding a learned baseline and engaging with the CTDE/literature would transform the interdisciplinary reach.

#### Readability for Nonspecialists: ★★★☆☆ (3/5 — Adequate but MARL-Opaque)

The manuscript is readable for roboticists but opaque for the MARL community. Key MARL concepts (CTDE, emergent communication, credit assignment, non-stationarity) are never mentioned. Conversely, UAV-domain concepts (G2-M2, C3, R_task) are undefined for the MARL reader. A glossary and explicit domain translation table would help.

### Key Questions for Authors

1. Why should a practitioner use your hand-designed VoI gate rather than training an IC3Net-style learned communication policy?
2. If CTDE-trained policies can achieve centralized performance without any execution-time communication, what is the remaining value of VoI-gated communication?
3. Would the VoI metric make a good *reward signal* for training a learned communication policy? (i.e., reward agents for communicating only when VoI is high)
4. Have you considered that your star topology (UAVs ↔ GCS) may be suboptimal compared to a learned peer-to-peer topology?

---

# Editor's Cross-Synthesis Report

## Manuscript: D-EPA-RHP Multi-UAV Redesign

**Three reviewers returned.** Overall pattern: strong conceptual framing, weak execution readiness. The central question — "how much communication is enough for multi-UAV teams?" — is timely, novel, and important. All three reviewers independently identified the marginal-benefit framing as the manuscript's primary contribution. However, all three also identified missing comparisons, underdeveloped theoretical connections, and insufficient infrastructure readiness.

---

### Cross-Dimensional Synthesis

| Dimension | R1 (Robotics) | R2 (Info Theory) | R3 (MARL) | **Consensus** |
|-----------|:------------:|:----------------:|:---------:|:------------:|
| Originality | ★★★★☆ | ★★★★☆ | ★★★☆☆ | **4/5** — Novel framing, VoI mechanism itself has precedents |
| Scientific Importance | ★★★★★ | ★★★★☆ | ★★★★☆ | **4.3/5** — Addresses acknowledged gaps across communities |
| Interdisciplinary Readership | ★★★★☆ | ★★★★★ | ★★★☆☆ | **4/5** — Exceptional potential, currently under-exploited |
| Technical Soundness | ★★★☆☆ | ★★★☆☆ | ★★☆☆☆ | **2.7/5** — Critical missing baselines and analyses across all three perspectives |
| Readability | ★★★☆☆ | ★★☆☆☆ | ★★★☆☆ | **2.7/5** — Acronym-heavy; lacks conceptual entry points for non-specialists |

**Aggregate score: 3.5/5 — Major Revision Required**

---

### Convergent Concerns (All Three Reviewers)

The three reviewers independently identified three overlapping gaps:

#### 1. Missing Baseline Types

| Missing Baseline | Raised By | Why Critical |
|-----------------|-----------|-------------|
| Fixed-rate periodic communication at equivalent bandwidth | R2, R3 | Cannot distinguish "VoI-gating is beneficial" from "any communication helps" |
| Learned communication (IC3Net-style gating) | R3 | Cannot claim model-based VoI advantage without learned comparison |
| Random-sampling communication | R1 | Cannot distinguish "VoI metric is informative" from "any selective scheme works" |
| CTDE-trained policy (zero execution communication) | R3 | If CTDE matches centralized, the communication problem is circumvented |

**Editor's recommendation**: Add at minimum (a) fixed-rate periodic communication and (b) one learned baseline. The former is a one-day experiment; the latter is more involved but essential for the MARL community's engagement.

#### 2. Underdeveloped Theoretical Connections

| Connection | Raised By | Action |
|-----------|-----------|--------|
| VoI gate = semantic event-triggered control (Zeno risk) | R2 | Analyze minimum inter-event time under C3 noise; add regularization if needed |
| PoA ≤ 2 bound may not apply (4.6× gap) | R1 | Explain or withdraw PoA argument |
| Witsenhausen dual role of communication | R2 | Explicitly frame VoI gate as nonlinear optimal policy approximation |
| AoII / semantic communication | R2 | Import AoII metrics; connect to semantic communication framework |
| CTDE as alternative to communication | R3 | Explain why CTDE is insufficient for this domain (dynamic tasks, heterogeneous capabilities) |

**Editor's recommendation**: Prioritize the Zeno analysis (R2) and the CTDE discussion (R3). The PoA issue (R1) should be resolved before claiming any theoretical protection.

#### 3. The Single-UAV Results as a Credibility Problem

All three reviewers noted — with varying degrees of concern — that the current single-UAV results (R_task = 0.29 vs Greedy = 0.79) undermine confidence in the multi-UAV extension. The gap of 4.6× exceeds the PoA bound of 2×, suggesting either the theory does not apply or the implementation is still buggy. Neither possibility inspires confidence.

**Editor's recommendation**: This is the most urgent issue to resolve before submission. Three options:
1. **Fix the implementation** until single-UAV R_task is within 50% of Greedy (PoA ≤ 2 bound), then proceed with multi-UAV
2. **Withdraw the PoA argument** and reframe: "we show that even when single-UAV distributed is poor, multi-UAV may still benefit from VoI-gating through task parallelization effects"
3. **Embrace the negative result**: "we demonstrate that distributed coordination is structurally disadvantaged regardless of team size — here is the information-theoretic bound that proves why"

Option 3 may yield the most impactful paper, following the Talamali et al. (2021, *Science Robotics*) precedent.

---

### Divergent Concerns (Reviewer-Specific)

**R1 (Robotics)** is most concerned about the experimental infrastructure not existing yet. This is a feasibility/gate concern — if the multi-UAV simulator cannot be built within a reasonable timeline, the paper is a proposal, not a result.

**R2 (Info Theory)** is most concerned about the lack of Zeno analysis and the absence of a fixed-rate baseline. These are theoretically critical but experimentally straightforward to address.

**R3 (MARL)** is most concerned about the absence of learned baselines. This is the most demanding revision request — adding a learned communication baseline is a non-trivial engineering effort. However, R3 also suggested the most constructive framing: "the VoI metric could serve as a reward signal for training learned communication policies" — a genuine interdisciplinary contribution.

---

### Recommended Revision Priorities

**Tier 1 (Must address before resubmission)**:
1. Resolve the single-UAV credibility gap (explain, fix, or reframe)
2. Add fixed-rate periodic communication baseline
3. Analyze VoI gate Zeno behavior under C3 noise; add regularization if needed
4. Specify seed × episode counts with statistical power justification

**Tier 2 (Strongly recommended)**:
5. Engage with learned communication literature (IC3Net, DCC, VIL2C) — even if not implementing a baseline, the manuscript must demonstrate awareness
6. Frame VoI gate within semantic communication / AoII framework
7. Add CTDE discussion: why CTDE-trained policies don't solve this problem
8. Decompose MSB(N) metric to separate UAV addition benefit from synchronization benefit

**Tier 3 (Would strengthen)**:
9. Add one learned communication baseline (IC3Net-style gating or MAPPO-CTDE)
10. Test whether adaptive θ_t outperforms fixed, tuned θ_t
11. Explicitly import AoI community metrics (peak AoI, AoII violation probability)
12. Add a conceptual overview diagram with plain-language labels

---

### Overall Editorial Verdict

**MAJOR REVISION — Fundamentally Promising**

This manuscript has a strong conceptual core: the marginal-benefit framing of multi-UAV information synchronization, systematically characterized across (N, B, T, C), is novel and important. The research question — "how much communication is enough?" — is timely and addresses gaps acknowledged by 2024–2025 survey papers across robotics, information theory, and MARL.

The primary weakness is execution readiness. The experimental infrastructure appears not to exist. Critical baselines are missing. The theoretical framing is underdeveloped relative to the communities the work claims to bridge. The single-UAV results undermine confidence in the multi-UAV extension.

However, each of these weaknesses is addressable. The revision path is clear and the required additions are well-defined. If the authors (a) resolve the single-UAV credibility issue, (b) add the three critical baselines, (c) develop the Zeno/AoII theoretical connection, and (d) engage minimally with the learned-communication literature, this manuscript would be competitive at IEEE T-RO, IJRR, or (for the strongest version) *Science Robotics*.

The editor notes that **the most impactful version of this paper may be the negative-result version**: "we characterise when and why multi-UAV distributed coordination fails to beat simple baselines, and prove an information-theoretic bound on when it must fail." This would follow the Talamali et al. (2021) precedent in *Science Robotics* and would be more original than a standard "our method beats baselines" paper. The authors are encouraged to consider whether their empirical results (if they align with the single-UAV pattern) are telling them something more interesting than they planned to find.

---

### Benchmarks: What This Work Should Be Compared Against

Based on the reviewers' comments and the editor's assessment, the following recent (2024–2026) papers constitute the appropriate comparison set. Note: CBBA (2009) is NOT appropriate as a primary benchmark — it addresses task allocation assuming free communication, while this work's contribution is precisely about communication as a scarce resource.

| Benchmark | Year | Venue | What It Does | Why Compare |
|-----------|------|-------|-------------|-------------|
| **RAG** (Xu & Tzoumas) | 2025 | IEEE T-RO | Distributed submodular optimization; analyzes approximation bound vs. neighborhood size | Closest existing work to marginal-benefit analysis; ~45 robots in AirSim |
| **AC-DMTA** | 2026 | MDPI Drones | Interval bids + consistency certificates; communication reduction vs. CBBA variants | State-of-the-art in communication-efficient task allocation; directly comparable |
| **ED-CBBA** (Sao et al.) | 2025 | IEEE ICUAS | Event-driven CBBA; 52% message reduction with zero quality loss | Demonstrates that most CBBA messages have zero marginal value — the same insight our VoI gate operationalizes |
| **Co-CoLoSSI** | 2024 | IEEE | Connectivity-aware planning; optimizes network topology for mission performance | Proactive communication management — complementary to our reactive VoI approach |
| **VIL2C** (Zhang et al.) | 2026 | AAAI | VoI-aware low-latency communication for MARL; KL-divergence of action distributions | Most direct comparison: uses VoI metric within learned framework vs. our model-based VoI gate |
| **Pan & Luo** | 2026 | arXiv | Information-theoretic phase transition: P(success) = O(2^(-Ω(C_min))) | Provides the theoretical framework for "when communication must fail" — our work could empirically validate this bound |
| **Talamali et al.** | 2021 | Science Robotics | "When less is more" — constrained communication improves swarm adaptability | Precedent for counterintuitive negative-result paper in top venue; similar framing |
| **Calvo-Fullana & How** | 2022 | ACC | VoI censoring for distributed Bayesian filtering; KL-divergence threshold | Applies VoI censoring to distributed estimation — similar mechanism, different domain |

---

## Post-Review Benchmark Update (2026-07-12)

A systematic search for 2024–2026 hot papers identified 15 papers across 5 categories. Key findings that strengthen or modify the review:

### Critical Benchmarks the Review Underweighted

**1. Zhu et al. (2024, *Science Robotics*) — "Self-organizing nervous systems for robot swarms"**

This is the highest-impact field-validated swarm paper of the past two years (Altmetric 99th percentile, 17 heterogeneous aerial-ground robots, simulation-scaled to 250+). The SoNS architecture ("locally centralized, globally distributed" hierarchy) is an alternative solution to the same problem: how to organize a swarm when global communication is unrealistic. Any new multi-UAV coordination method must articulate how it differs from SoNS's hierarchical self-organization approach.

**2. Ma et al. (2024, *Nature Machine Intelligence*) — "Efficient and scalable RL for large-scale network control"**

This is the highest-impact MARL-with-communication-constraints paper, published in a Nature-family journal. MDPO achieves coordination with only ~5–35% of centralized communication cost, validated on real-world networks up to 436 agents. The bar for "communication efficiency" claims is now defined by this paper.

**3. Arul et al. (2024, *IROS*) — "When, What, and with Whom to Communicate"**

The closest structural match to D-EPA-RHP's VoI-gating for multi-robot coordination. Uses a visual transformer + self-attention to learn selective communication. Achieves ~2× improvement in navigation success and ~20% reduction in path length vs. full-broadcast baselines. This is the direct learned-communication baseline that Reviewer 3 demanded.

**4. Zhu et al. (2024, *JAAMAS*) — "A survey of multi-agent DRL with communication"**

Must-cite taxonomic reference (~155 citations). Defines 9 analytical dimensions for evaluating communication in MARL. Any new communication method should position itself using this taxonomy.

### Implications for the Review

- **R3's demand for a learned baseline is validated.** Arul et al. (IROS 2024) is the exact comparison point. The manuscript should at minimum discuss how VoI-gating differs from their learned "when, what, with whom" approach.
- **The bar for field validation is now defined by SoNS** (17 real robots, 7 years of development). The manuscript does not need to match this — but should acknowledge that simulation-only validation is a limitation relative to the state of the art.
- **MDPO (Nature Machine Intelligence 2024) redefines the scalability claim.** Any claim about "scalable distributed coordination" must be calibrated against MDPO's 436-agent real-world validation.
- **The key gap identified by the benchmark search confirms our contribution space.** No existing paper simultaneously combines: VoI-driven communication scheduling + distributed task allocation + receding-horizon path planning + degraded communication channels in a single framework. The individual components exist — the integration is novel.

### Updated Benchmark Priority for Manuscript

| Must-compare | Should-cite | Contextual reference |
|-------------|-------------|---------------------|
| Arul et al. (IROS 2024) | Zhu et al. (JAAMAS 2024) | Zhu et al. (Sci Robotics 2024) |
| Xu & Tzoumas (IEEE T-RO 2025) | Ma et al. (Nat Mach Intell 2024) | Li et al. (NeurIPS 2024) |
| Zhang et al. / VIL2C (AAAI 2026) | Chiariotti & Fabris (2025) | Ding et al. (NeurIPS 2024) |
| Sao et al. / ED-CBBA (ICUAS 2025) | Emami et al. (IEEE TVT 2024) | Luna et al. (JFR 2024) |

---

*Review conducted 2026-07-12. Three reviewers + editor synthesis. All reviewers declare no conflict of interest.*
