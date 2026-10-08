# Evaluation notes for Wald-Q4B v2 (`04701-c22`)

- **Decision Index.** Every training file of all three stages was scanned against the complete Decision Index 0.3 public suite with our strict matcher: **0 strict hits** ([contamination/trained-on-di-ids.json](contamination/trained-on-di-ids.json): ids, rule, file hashes, what is not covered). Each stage's data had also been gated before training (stage 1 against the 0.2 suite, stage 3 against the 0.3 suite with additional exact full-shingle and character n-gram checks; every hit row was dropped). The training data contains train splits of some benchmarks whose test items the Decision Index also uses ([PROVENANCE.md](PROVENANCE.md)), and the model was developed with the index's task formats in view, so its Decision Index results are not held-out results in the usual sense. The organizer's private tests are unknown to us.
- **JevBench public set (231).** A development scoreboard: never training data, but read repeatedly during development. Not held out.
- **JevBench-XL** (internal). Stage 2's and stage 3's new items were exact-state deduplicated against all XL partitions, JevBench, JevAdvBench and the Decision Index sample; the XL TEST partitions were not used for any selection. The XL scores are internal and not independently verifiable.
- **multistep_decisions (100)** and **JevAdvBench** are evaluation-only.
- **Checkpoint and policy selection.** Auto 0.7 was fixed as the default before the Decision Index 0.3 runs. The released checkpoint (step 300 of 638) was chosen over the final step and over the earlier C16B / C16C checkpoints using paired screens on a fixed subset of the Decision Index 0.3 public suite and then one full public run; the public index therefore informed the choice of checkpoint. No successful request was repeated.
- **Calibration.** One frozen temperature table (`temperature.json`, sha256 `a0f72cd2d0a653e81051e5a0c77fc1a69131552a8102b580a93a6dbe7908b2da`), fitted on 275 development rows before these evaluations; no benchmark labels were used, and the Decision Index 0.3 run did not refit it.
- **Vision.** The vision tower is Qwen3.5-4B's own, unchanged; no image data was used in training. Image results are zero-shot and depend on the base model's pretraining, which may include the public image benchmarks.
- **Not ruled out:** semantic overlap, and contamination in the base model's pretraining data.
- No request payloads, gold labels or generated reasoning text are included in this repository.

The notes for v1.x are at their tags.
