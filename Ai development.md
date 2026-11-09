# Ai development

Comprehensive research summary of AI model development from **DeepSeek**, **GLM (Zhipu AI)**, and **Kimi (Moonshot AI)**.

Compiled from each lab's GitHub repositories, technical reports, and model cards.

---

## 1. DeepSeek (deepseek-ai)

DeepSeek is a Chinese AI lab known for open-weight Mixture-of-Experts (MoE) language models and strong reasoning capabilities.

### DeepSeek-V3
- **Source:** `github.com/deepseek-ai/DeepSeek-V3`
- **License:** MIT
- **Architecture:** Mixture-of-Experts (MoE), 671B total parameters, 37B activated per token
- **Context length:** 128K tokens
- **Key innovations:**
  - Multi-head Latent Attention (MLA)
  - DeepSeekMoE architecture (validated in DeepSeek-V2)
  - Auxiliary-loss-free load balancing strategy
  - Multi-token prediction (MTP) training objective (also used for speculative decoding)
  - FP8 mixed-precision training framework, first validated at extremely large scale
- **Training:** Pre-trained on 14.8 trillion tokens using ~2.788M H800 GPU hours (full training); no irrecoverable loss spikes or rollbacks
- **Post-training:** Knowledge distillation from DeepSeek-R1 (reasoning/verification/reflection patterns) via Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL)
- **Variants:** DeepSeek-V3-Base, DeepSeek-V3 (chat/instruct), plus 14B MTP module weights
- **Notable benchmarks:** MMLU 87.1, MMLU-Pro 64.4, HumanEval 65.2, MATH 61.6, LiveCodeBench 19.4 — outperforms other open-source models, comparable to leading closed-source models

### DeepSeek-R1
- **Source:** `github.com/deepseek-ai/DeepSeek-R1`
- **License:** MIT
- **Models:** DeepSeek-R1-Zero, DeepSeek-R1, plus six dense distilled models (1.5B, 7B, 8B, 14B, 32B, 70B) based on Llama and Qwen
- **Base:** Trained on DeepSeek-V3-Base
- **Context length:** 128K
- **Key innovation:** First open research to demonstrate that reasoning capabilities can be incentivized purely through large-scale Reinforcement Learning (RL) without Supervised Fine-Tuning (SFT) as a preliminary step (DeepSeek-R1-Zero)
- **Training pipeline:** Two RL stages (improved reasoning patterns + human preference alignment) and two SFT stages (seed for reasoning and non-reasoning capabilities)
- **Distillation:** Reasoning patterns from the large model distilled into smaller dense models; DeepSeek-R1-Distill-Qwen-32B outperforms OpenAI-o1-mini across benchmarks
- **Performance:** Comparable to OpenAI-o1 across math, code, and reasoning tasks

---

## 2. GLM / Zhipu AI (zai-org / Zhipu AI)

Zhipu AI develops the GLM family of open multilingual multimodal chat language models.

### GLM-4 Series
- **Source:** `github.com/zai-org/GLM-4`
- **License:** Apache-2.0
- **Website:** z.ai (free experience), bigmodel.cn (commercial API)
- **Technical report:** Released 2024/06/18

### GLM-4-32B-0414 (2025/04/14)
- **Parameters:** 32 billion
- **Pre-training:** 15 trillion high-quality tokens, including substantial reasoning-type synthetic data
- **Post-training:** Human preference alignment for dialogue; rejection sampling and RL for instruction following, code, function calling
- **Capabilities:** Engineering code, Artifact generation, function calling, search-based Q&A, report generation; comparable to GPT-4o and DeepSeek-V3-0324 (671B) on several benchmarks
- **Variants:** GLM-4-32B-Base-0414, GLM-4-32B-0414

### GLM-Z1-32B-0414
- **Type:** Reasoning model with deep thinking
- **Training:** Cold start, extended RL, further training on mathematics, code, and logic; general RL with pairwise ranking feedback
- **Improvement:** Significantly improved math and complex task solving vs. base model

### GLM-Z1-Rumination-32B-0414
- **Type:** Deep reasoning model with rumination capabilities (against OpenAI's Deep Research)
- **Distinction:** Deeper and longer thinking for open-ended, complex problems (e.g., comparative analysis)
- **Training:** Scaling end-to-end RL with responses graded by ground-truth answers or rubrics; can use search tools during deep thinking

### GLM-4.1V (2025/07/02)
- **Type:** GLM-4.1V-9B-Thinking series VLM (Vision-Language Model)

---

## 3. Kimi / Moonshot AI (moonshotai)

Moonshot AI is committed to solving ambitious "moonshot" problems toward AGI, with a strong open-source contribution record.

### Kimi K3 (2025/07/27)
- **Paper:** arXiv:2607.24653, "Kimi K3: Open Frontier Intelligence"
- **Parameters:** 2.8T total (world's first open 3T-class model)
- **Architecture:** Kimi Delta Attention (KDA) and Attention Residuals (AttnRes), native vision capabilities
- **Context window:** 1 million tokens
- **Focus:** Frontier intelligence across long-horizon coding, knowledge work, and reasoning

### Attention Residuals
- Drop-in replacement for residual connections with consistent scaling gains

### Kimi K2.5
- **Paper:** arXiv:2602.02276, "Kimi K2.5: Visual Agentic Intelligence" (2025/02/02)
- **Type:** Open-source, native multimodal agentic model
- **Training:** Continual pretraining on ~15 trillion mixed visual and text tokens atop Kimi-K2-Base
- **Variants:** Kimi-K2.5, Kimi-K2.6, Kimi-K2.7-Code (1T class)

### Kimi K2
- **Type:** Open-source Mixture-of-Experts model
- **Parameters:** 32B activated, 1T total
- **Focus:** State-of-the-art performance in frontier knowledge, math, and coding among non-thinking models

### Kimi-K1.5
- **Paper:** "Scaling Reinforcement Learning with LLMs"
- **Type:** o1-level multi-modal reasoning model tech report

### Kimi-VL
- **Type:** Mixture-of-Experts Vision-Language Model for multimodal reasoning, long-context understanding, and strong agent capabilities
- **Variant:** Kimi-VL-A3B-Thinking / Kimi-VL-A3B-Thinking-2506 (16B)

### Kimi-Audio
- **Type:** Universal audio foundation model
- **Capabilities:** Speech recognition, audio understanding, audio-to-text chat, speech-to-speech conversation
- **Release:** Technical report, model weights, and evaluation toolkit

### Kimina-Prover Preview
- **Type:** Large formal reasoning models with Reinforcement Learning
- **Release:** Technical report, model weights, rectified miniF2F-test benchmark

### Supporting research & infrastructure
- **Muon:** "Muon is Scalable for LLM Training" — improved Muon optimizer + checkpoints for SOTA small models
- **MoBA:** Mixture of Block Attention for Long-Context LLMs — code and paper for long-context technique
- **MoonEP:** Expert parallelism communication library; perfectly balanced token loads via dynamic redundant experts
- **FlashKDA:** High-performance Kimi Delta Attention kernels built on CUTLASS
- **Mooncake:** KV-centric disaggregated LLM serving; Best Paper at FAST 2025
- **Mooncake / Moonpalace:** Debugging tools for API service

### Benchmarks (Moonshot AI)
- **WorldVQA:** Benchmark for atomic vision-centric world knowledge in MLLMs
- **PerceptionBench:** Benchmark for atomic visual perception in MLLMs
- **CombiBench:** First benchmark for combinatorial mathematics problems formalized in Lean 4

### Agents
- **Kimi Code:** Fast, versatile, extensible AI coding agent bringing Kimi into development workflows (CLI)

---

## Summary Comparison

| Lab | Flagship Models | Key Architecture | Notable Innovation |
|-----|----------------|------------------|-------------------|
| **DeepSeek** | V3, R1, R1-Zero, distilled dense models | MoE (MLA, DeepSeekMoE), 671B total / 37B activated | Pure-RL reasoning (R1-Zero), auxiliary-loss-free MoE, FP8 training, MTP |
| **GLM (Zhipu)** | GLM-4-32B, GLM-Z1, GLM-Z1-Rumination, GLM-4.1V | Dense multilingual multimodal | 15T token pre-training, rumination/deep thinking RL, search-augmented reasoning |
| **Kimi (Moonshot)** | Kimi K3, K2.5, K2, K1.5, Kimi-VL, Kimi-Audio | MoE, Kimi Delta Attention (KDA), Attention Residuals | 2.8T-param 3T-class model, 1M-token context, Muon optimizer, MoBA, Mooncake disaggregated serving |

---

---

## 4. Newer Flash & Efficient Models (for training-catalogue reference)

### GLM 5.3 / GLM 5.3-Flash (Zhipu AI / zai-org)
- **Source:** `github.com/zai-org/GLM-5` (blog: z.ai/blog/glm-5.3)
- **License:** Apache-2.0 (GLM-5 repo); GLM-5.3-Flash HF card lists MIT
- **Positioning:** GLM-5.3 uses the same base model as GLM-5.2; every gain comes from post-training. Strongest open-weights model for coding, ~50% improvement over GLM-5.2 on the in-house Z.ai Code Bench; open-source SOTA on Terminal Bench 3.0 and Agents' Last Exam; state of the art on CyberGym for vulnerability discovery.
- **GLM-5.3-Flash:** First natively multimodal model in the GLM-5 series (text, image, video). Released August 26, 2026. **320B total parameters, 18B active per token** (hybrid MoE), **1,048,576 (1M) token context window**, native **FP8 (e4m3, dynamic scaling)** precision. Architecture: **34 KDA linear-attention layers interleaved with 11 DSA sparse-attention layers** (3:1 ratio); DSA/MLA layers run in pure NoPE mode. Introduces a **hybrid architecture combining sparse and linear attention**, sharply reducing long-context serving costs while preserving precise long-context capabilities. Adopts **Manifold-Constrained Hyper-Connections (mHC)** for scaling efficiency. Trained on a **30T-token multimodal pre-training corpus**. Delivers more intelligence with less compute — roughly one-tenth the price of GLM-5.2, approaching Claude Opus 4.8 on coding and agentic benchmarks.
- **Related:** GLM-5.2 (1M-token context, IndexShare sparse-attention indexer reducing per-token FLOPs 2.9× at 1M context, improved MTP layer), GLM-5.1 (long-horizon agentic engineering), GLM-5 (complex systems engineering).
- **HF:** `zai-org/GLM-5.3-Flash` (321B parameters listed on HF model card)

### DeepSeek V4.1 Flash
- **Source:** deepseek.com/en/news/deepseek-v4-1-flash/ (Sept 10, 2026); HF `deepseek-ai/DeepSeek-V4.1-Flash-1017`
- **License:** MIT
- **Version:** DeepSeek V4.1-Flash-1017
- **Positioning:** Smallest model in the new V4 architecture family, with native visual understanding. Designed for greater capability, faster inference, higher throughput, and scaling to larger models. API model name: `deepseek-flash`. Replaces V4-Flash and V4-Flash-Vision-Exp; V4-Pro is being phased out (routes to V4.1-Flash from Sept 14, 2026).
- **Architecture:** **552B-parameter MoE** with an **asymmetric Causal Encoder–Decoder architecture**: just **8B active parameters for input, 16B for output** (more intelligence, less cost). New pretraining methods + larger-scale RL post-training deliver benchmark results ahead of flagship models including DeepSeek-V4-Pro.
- **Efficiency:** KV cache needs just **1/4 the HBM** and **1/8 the SSD storage** vs. previous generation — significant for agent workloads where cache-hit charges dominate cost.
- **Benchmarks (HF model card):** Terminal-Bench 42.3, DeepSWE 77, Agents' Last Exam 35.4 overall pass rate, HLE 42.6 (without tools), 44 (text-only subset).
- **Native multimodal support.** FP8. Off-peak API pricing at 50% of peak rates.

### DeepSeek smaller, more efficient models (training-data candidates)

#### DeepSeek-V2-Lite
- **Source:** `github.com/deepseek-ai/DeepSeek-V2` (news: DeepSeek-V2-Lite released 2024-05-16)
- **License:** MIT
- **Parameters:** 16B total, 2.4B activated per token
- **Context length:** 32K tokens
- **Architecture:** MLA + MoE (Multi-head Latent Attention + Mixture-of-Experts)
- **Training:** Pretrained on 8.1 trillion tokens, followed by SFT and RL
- **Economy:** vs. DeepSeek-V2 (236B/21B), far smaller/cheaper while still outperforming larger dense models on several benchmarks (e.g., MMLU 58.3 vs. LLaMA3 70B 48.2; C-Eval 81.7 vs. LLaMA3 70B 67.5)
- **Variants:** DeepSeek-V2-Lite, DeepSeek-V2-Lite-Chat (SFT)

#### DeepSeek-Coder-V2-Lite
- **Source:** `github.com/deepseek-ai/DeepSeek-Coder-V2`
- **License:** MIT
- **Parameters:** 16B total, 2.4B activated per token
- **Context length:** 128K tokens
- **Training:** Further pre-trained from an intermediate checkpoint of DeepSeek-V2 with an additional 6 trillion tokens (code-focused)
- **Capabilities:** Code intelligence comparable to GPT-4-Turbo / Claude 3 Opus / Gemini 1.5 Pro on code and math benchmarks; 338 programming languages supported; HumanEval 81.1 (Lite-Instruct), MBPP+ 68.8
- **Variants:** DeepSeek-Coder-V2-Lite-Base, DeepSeek-Coder-V2-Lite-Instruct

#### DeepSeek-V2 (efficient flagship reference)
- **Source:** `github.com/deepseek-ai/DeepSeek-V2`; arxiv:2405.04434
- **License:** MIT
- **Parameters:** 236B total, 21B activated per token; 128K context
- **Key efficiency gains:** 42.5% lower training cost than DeepSeek 67B, 93.3% KV-cache reduction, 5.76× higher max generation throughput; MLA + DeepSeekMoE; FP8-friendly efficient training
- **Variants:** DeepSeek-V2, DeepSeek-V2-Chat (RL)

---

## Summary Comparison — All Labs

| Lab | Flagship / Flash models | Efficient small models | Key efficiency techniques |
|-----|------------------------|-----------------------|---------------------------|
| **DeepSeek** | V4.1-Flash-1017 (552B MoE, 8B/16B active), V3, R1 | V2-Lite (16B/2.4B), Coder-V2-Lite (16B/2.4B) | MLA, DeepSeekMoE, asymmetric encoder-decoder, auxiliary-loss-free MoE, FP8, MTP, KV-cache reduction |
| **GLM (Zhipu)** | GLM-5.3, GLM-5.3-Flash (320B/18B, 1M context, FP8) | GLM-4.1V-9B (9B VLM thinking) | Sparse + linear attention hybrid, 34 KDA + 11 DSA layers, mHC, IndexShare (2.9× FLOPs reduction), MTP speculative decoding |
| **Kimi (Moonshot)** | Kimi K3 (2.8T), K2.5, K2 | Kimi-VL-A3B (16B), Moonlight-16B | Kimi Delta Attention (KDA), Attention Residuals, MoBA, FlashKDA kernels |

---

**Sources:**
- DeepSeek-V3: https://github.com/deepseek-ai/DeepSeek-V3
- DeepSeek-R1: https://github.com/deepseek-ai/DeepSeek-R1
- DeepSeek-V2 (incl. V2-Lite): https://github.com/deepseek-ai/DeepSeek-V2
- DeepSeek-Coder-V2 (incl. Coder-V2-Lite): https://github.com/deepseek-ai/DeepSeek-Coder-V2
- DeepSeek models page: https://www.deepseek.com/models
- DeepSeek V4.1 Flash news: https://www.deepseek.com/en/news/deepseek-v4-1-flash/
- DeepSeek V4.1-Flash-1017 (HF): https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017
- DeepSeek (Wikipedia): https://en.wikipedia.org/wiki/DeepSeek
- GLM-5 (incl. 5.3 & 5.3-Flash): https://github.com/zai-org/GLM-5
- GLM-5.3-Flash blog: https://z.ai/blog/glm-5.3-flash
- GLM-5.3-Flash (HF): https://huggingface.co/zai-org/GLM-5.3-Flash
- GLM-5 (incl. 5.3 & 5.3-Flash): https://github.com/zai-org/GLM-5
- GLM-V (4.1V/4.5V/4.6V): https://github.com/zai-org/GLM-V
- GLM-4: https://github.com/zai-org/GLM-4
- Moonshot AI (GitHub): https://github.com/moonshotai
- Moonshot AI (Hugging Face): https://huggingface.co/moonshotai

