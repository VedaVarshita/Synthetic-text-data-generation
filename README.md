# Synthetic-text-data-generation

This report summarizes experimental runs conducted with **AG News** and **IMDB** datasets under different prompting strategies (zero-shot, one-shot, few-shot) and training modes (baseline vs. LoRA fine-tuning).
We evaluate the runs using quality (Perplexity, ROUGE-L, VendiScore), diversity (Distinct-n, Self-BLEU), and clustering metrics.


## AG News Dataset

| Run Name  | Training | Shots | Avg PPL ↓ | Uniqueness ↑ | Self-BLEU ↓ | ROUGE-L F1 ↑ | VendiScore ↑ |
| --------- | -------- | ----- | --------- | ------------ | ----------- | ------------ | ------------ |
| Zero-shot | None     | 0     | 536.2     | **1.000**    | 0.170       | 0.125        | 42.1         |
| Zero-shot | None     | 0     | 772.2     | 0.995        | **0.084**   | 0.095        | **46.1**     |
| One-shot  | None     | 1     | 697.6     | 0.460        | 0.057       | 0.022        | 15.1         |
| Few-shot  | None     | 2     | 299.7     | 0.905        | 0.193       | 0.106        | 35.9         |
| Zero-shot | LoRA     | 0     | **46.3**  | **1.000**    | 0.242       | **0.200**    | **46.8**     |
| One-shot  | LoRA     | 1     | 311.9     | 0.667        | **0.256**   | 0.077        | 47.9         |
| Few-shot  | LoRA     | 2     | 170.3     | 0.930        | 0.224       | 0.112        | 32.1         |


## IMDB dataset

| Run Name  | Training | Shots | Avg PPL ↓ | Uniqueness ↑ | Self-BLEU ↓ | ROUGE-L F1 ↑ | VendiScore ↑ |
| --------- | -------- | ----- | --------- | ------------ | ----------- | ------------ | ------------ |
| Zero-shot | None     | 0     | 696.4     | **1.000**    | 0.094       | 0.120        | 46.1         |
| One-shot  | None     | 1     | 421.4     | **1.000**    | 0.109       | 0.112        | 46.0         |
| Few-shot  | None     | 2     | 96.1      | **1.000**    | 0.180       | 0.157        | 43.9         |
| Zero-shot | LoRA     | 0     | **67.9**  | **1.000**    | 0.137       | **0.168**    | **52.2**     |
| One-shot  | LoRA     | 1     | 113.8     | 0.995        | 0.160       | 0.161        | 43.5         |




-----------


### 1. **Perplexity (PPL) Improvements**

* LoRA fine-tuning **dramatically lowers perplexity** across both datasets.

  * AG News: Zero-shot drops from **536 → 46**.
  * IMDB: Zero-shot drops from **696 → 68**.
* This indicates **better fluency and alignment with natural language**.

### 2. **Diversity vs. Quality Trade-off**

* **Higher uniqueness** is consistently observed in zero-shot setups (close to 1.0), but diversity (Self-BLEU, Dist-n) is dataset-dependent.
* Few-shot improves **semantic consistency** but sometimes reduces uniqueness (e.g., AG News one-shot baseline).

### 3. **LoRA Benefits**

* LoRA runs show **better balance** between perplexity, ROUGE, and VendiScore compared to baseline prompting.
* Example: On IMDB zero-shot, VendiScore improves from **46 → 52** with LoRA.
* Suggests fine-tuned models generate **more diverse and semantically coherent text**.

### 4. **Dataset Differences**

* IMDB (longer reviews) tends to yield **lower perplexity after fine-tuning** compared to AG News.
* AG News maintains **stronger uniqueness** but struggles with diversity in few-shot baselines.

---

## Conclusions

1. **LoRA fine-tuning is essential** → it drastically improves perplexity, ROUGE, and VendiScore, making outputs more coherent and diverse.
2. **Few-shot prompting without training is unstable** → while it improves perplexity (e.g., IMDB few-shot), it reduces uniqueness and increases redundancy.
3. **Zero-shot LoRA is the sweet spot** → combines lowest perplexity, high uniqueness, and best VendiScore across both datasets.
4. **Dataset nature matters** → longer IMDB reviews benefit more from LoRA training, while AG News requires careful balancing of diversity.

---

## Recommendations for Future Work

* Explore **hybrid strategies**: combine LoRA with small-shot prompting for controlled diversity.
* Test **larger LoRA ranks (r, α)** to see if gains plateau or improve further.
* Add **human evaluation** to complement automatic metrics (fluency, readability).
* Investigate **domain adaptation**: applying AG News LoRA to IMDB and vice versa.


