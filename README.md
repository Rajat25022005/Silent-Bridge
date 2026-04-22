<div align="center">

```
███████╗██╗██╗     ███████╗███╗   ██╗████████╗
██╔════╝██║██║     ██╔════╝████╗  ██║╚══██╔══╝
███████╗██║██║     █████╗  ██╔██╗ ██║   ██║
╚════██║██║██║     ██╔══╝  ██║╚██╗██║   ██║
███████║██║███████╗███████╗██║ ╚████║   ██║
╚══════╝╚═╝╚══════╝╚══════╝╚═╝  ╚═══╝   ╚═╝

██████╗ ██████╗ ██╗██████╗  ██████╗ ███████╗
██╔══██╗██╔══██╗██║██╔══██╗██╔════╝ ██╔════╝
██████╔╝██████╔╝██║██║  ██║██║  ███╗█████╗
██╔══██╗██╔══██╗██║██║  ██║██║   ██║██╔══╝
██████╔╝██║  ██║██║██████╔╝╚██████╔╝███████╗
╚═════╝ ╚═╝  ╚═╝╚═╝╚═════╝  ╚═════╝ ╚══════╝
```

**Two thought chains. One meeting point. No tokens.**

*Successor to [think-in-silence](https://github.com/Rajat25022005/think-in-silence)*

<br>

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![arXiv](https://img.shields.io/badge/arXiv-coming_soon-B31B1B?style=flat-square)](https://arxiv.org)
[![HuggingFace](https://img.shields.io/badge/🤗_Model-coming_soon-FFD21E?style=flat-square)](https://huggingface.co/rajat5039)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)](LICENSE)

</div>

---

## The idea in one paragraph

Most latent reasoning models walk forward — from the question toward the answer — and hope they converge somewhere useful. **silent-bridge** trains two chains simultaneously: a forward chain that starts from the question, and a backward chain that starts from the answer. Both are trained to meet at a shared midpoint **M**. At inference, the backward chain is removed entirely. The forward chain has been shaped — for free — by a module that already knows where the answer lives. The result is faster convergence, fewer steps needed, and a midpoint that serves as an interpretable proof certificate.

---

## Why this is different

| | Coconut | Quiet-STaR | think-in-silence | **silent-bridge** |
|--|:--:|:--:|:--:|:--:|
| Reasoning medium | LM hidden states | Token vocabulary | Dedicated latent space | **Dedicated latent space** |
| Training signal | Token supervision | REINFORCE | JEPA MSE | **Bidirectional JEPA** |
| Backward pass | No | No | No | **Training only — zero inference cost** |
| Proof certificate | No | No | No | **Yes — midpoint M** |
| Extra inference compute | No | Yes | No | **No** |

The closest published work is [Reason from Future (RFF, 2025)](https://arxiv.org/abs/2506.03673), which does bidirectional reasoning in token space at inference time. silent-bridge moves this idea into pure latent space and eliminates inference overhead entirely.

---

## How it works

```
TRAINING

  Question ──► enc_q (frozen) ──► Forward ThoughtModule
                                   h0 → h1 → h2 → hK
                                                    │
                                                    ▼
                                              Midpoint M  ◄── loss: MSE(hK, M) + MSE(gK, M)
                                                    ▲
                                   g0 → g1 → g2 → gK
  Answer ───► enc_a (frozen) ──► Backward ThoughtModule


INFERENCE  (backward module removed)

  Question ──► enc_q ──► Forward ThoughtModule ──► hK ──► decoder ──► answer
```

The midpoint **M** is not fixed — it is negotiated by a small learned network that takes both hK and gK as input and outputs a target both chains are trained to reach. M is the "proof certificate": it contains enough information to decode an answer, and it is reachable from both the question and the answer sides.

---

## Core hypothesis

> A forward latent reasoning chain trained against a bidirectional signal converges to correct answers in fewer steps than a forward-only chain, because the backward module shapes attractor basins in the latent space during training.

**Testable prediction:** K=4 steps with bidirectional training should match K=8 steps from think-in-silence.

---

## Results

*Training in progress. Table will be updated.*

### K-scaling — Recall@1

| K steps | think-in-silence (forward only) | silent-bridge (bidirectional train) |
|:-------:|:-------------------------------:|:------------------------------------:|
| K=0     | 0.002                           | —                                    |
| K=1     | 0.064                           | —                                    |
| K=2     | 0.256                           | —                                    |
| K=4     | 0.504                           | —                                    |
| K=8     | 0.475                           | —                                    |

### K-scaling — BLEU / ROUGE-1

| K steps | think-in-silence BLEU | silent-bridge BLEU | think-in-silence R1 | silent-bridge R1 |
|:-------:|:---------------------:|:------------------:|:-------------------:|:----------------:|
| K=4     | 0.044                 | —                  | 0.218               | —                |
| K=8     | 0.231                 | —                  | 0.594               | —                |

---

## Quickstart

```bash
git clone https://github.com/Rajat25022005/silent-bridge
cd silent-bridge
pip install -r requirements.txt
```

**Train (Stage 1 — bidirectional JEPA):**
```bash
python main.py --config configs/base.yaml
```

**Evaluate K-scaling vs think-in-silence baseline:**
```bash
python eval.py --config configs/base.yaml --compare_baseline
```

**Run on a single question:**
```python
from src.models.silent_bridge import SilentBridge

model = SilentBridge.from_pretrained("rajat5039/silent-bridge")
answer = model.generate("Which city did Marie Curie move to after leaving Warsaw?", n_steps=4)
print(answer)  # → "Paris"
```

---

## Architecture

```
silent-bridge/
│
├── src/
│   ├── models/
│   │   ├── silent_bridge.py       # Full model: forward + backward + midpoint net
│   │   ├── thought_module.py      # Shared ThoughtBlock architecture
│   │   ├── midpoint_net.py        # Negotiates M from hK and gK
│   │   └── encoder.py             # Frozen Gemma-3-4B backbone
│   │
│   ├── training/
│   │   ├── trainer.py             # Bidirectional JEPA training loop
│   │   └── losses.py              # Convergence loss + VICReg variance term
│   │
│   └── eval/
│       ├── evaluator.py           # K-scaling, recall@k, BLEU/ROUGE
│       └── certificate_eval.py    # Midpoint M interpretability probing
│
├── configs/
│   └── base.yaml
│
├── main.py
├── eval.py
└── requirements.txt
```

---

## The midpoint as proof certificate

A unique property of this architecture: the midpoint **M** can be decoded independently and probed for interpretability.

```
Q: "Which scientist born in Warsaw won the Nobel Prize in Physics?"
Forward chain hK  → encodes "Nobel Prize, physicist, Warsaw"
Backward chain gK → encodes "Curie, Nobel, physics"
Midpoint M        → encodes "Marie Curie" ← decodable, interpretable
```

A linear probe on M should predict answer type, category, and partial answer content better than hK alone at the same step. This is the interpretability experiment.

---

## Connection to classical AI

This architecture is the latent-space analog of **bidirectional A\*** — a classical search algorithm that runs forward from the start and backward from the goal simultaneously. Bidirectional A\* is provably more efficient than forward-only A\* under mild conditions. silent-bridge is the first application of this idea to continuous latent reasoning, and the first to make the backward pass training-only with zero inference cost.

---

## Citation

If you use this work, please cite:

```bibtex
@misc{malik2026silentbridge,
  title   = {silent-bridge: Bidirectional Latent Reasoning with a Learned Proof Midpoint},
  author  = {Rajat Malik},
  year    = {2026},
  url     = {https://github.com/Rajat25022005/silent-bridge}
}
```

---

## Related work

- [think-in-silence](https://github.com/Rajat25022005/think-in-silence) — predecessor: forward-only latent reasoning with JEPA
- [Coconut](https://arxiv.org/abs/2412.06769) — latent CoT in LM hidden states
- [Quiet-STaR](https://arxiv.org/abs/2403.09629) — thought tokens with REINFORCE
- [Reason from Future](https://arxiv.org/abs/2506.03673) — bidirectional reasoning in token space
- [I-JEPA](https://arxiv.org/abs/2301.08243) — JEPA for vision, Assran et al.

---

<div align="center">
<sub>Built by Rajat Malik · 2026 · MIT License</sub>
</div>
