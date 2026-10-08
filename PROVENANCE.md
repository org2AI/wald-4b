# Training data of Wald-Q4B v2 (checkpoint `04701-c22`)

Base model: [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) (chat, revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`), Apache-2.0. Wald-Q4B v2 was trained from it by full-parameter decision training.

**Data.** About 15% of training tokens come from the train splits of public benchmarks that the Decision Index also draws on, reformatted as decisions. The rest is broad decision data: general decision tasks — rules and policies, classification, entailment and fact checking, tables and documents, web and tool actions, judging (about 40%); math and step-by-step reasoning (about 20%); and hard decisions written and/or labelled by large frontier models, including robustness cases with perturbed or adversarial states (about 25%). Public datasets are used as train splits only. Contamination scan against the full Decision Index 0.3 public suite: 0 strict hits.

**Contamination.** Every training file was scanned with a strict state / option matcher against the complete Decision Index 0.3 public suite: 0 hits. Details: [CONTAMINATION.md](https://huggingface.co/org2ai/Wald-4B/blob/main/CONTAMINATION.md).
