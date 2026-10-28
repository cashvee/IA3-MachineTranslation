# English-to-Kannada Machine Translation

**Course:** ICT4403 — Foundations of Generative AI  
**Assignment:** Assignment 3  
**Task:** English → Kannada Neural Machine Translation

## Overview

This project explores neural machine translation from English to Kannada using four different model approaches:

1. **Basic Transformer** — a character-level encoder–decoder Transformer trained from scratch.
2. **mT5-small** — a pretrained multilingual encoder–decoder model fine-tuned for translation.
3. **TinyLlama-1.1B** — a decoder-only language model evaluated using prompting, without fine-tuning.
4. **Gemma-2-2B-IT** — a decoder-only instruction-tuned language model evaluated using prompting, without fine-tuning.

The project compares translation quality, model size, training cost, inference speed, and the effect of generation temperature. It also examines translation errors and responsible AI considerations.

All experiments are conducted in **Kaggle Notebooks** using GPU acceleration where available.

## Objectives

- Build and evaluate a Transformer-based translation model from scratch.
- Fine-tune a pretrained multilingual model for English-to-Kannada translation.
- Investigate prompt-only translation using decoder-only language models.
- Compare BLEU and chrF scores on a shared evaluation set.
- Measure training time, inference time, and parameter count.
- Study the effect of generation temperature on translation quality.
- Identify limitations involving hallucination, tokenization, bias, and cultural nuance.

## Dataset

**Source:** [English to Kannada Machine Translation Dataset — Kaggle](https://www.kaggle.com/datasets/parvmodi/english-to-kannada-machine-translation-dataset)

The dataset contains parallel English and Kannada text stored in aligned files:

- `train.en` — English source sentences
- `train.kn` — corresponding Kannada translations

The original files contain approximately 4.09 million lines each.

To keep experimentation computationally manageable, a reservoir sample of 20,000 aligned sentence pairs is used.

| Split | Number of pairs |
|---|---:|
| Training | 16,000 |
| Validation | 2,000 |
| Test | 2,000 |
| **Total** | **20,000** |

The main model comparison uses the same 200-sentence evaluation subset from the held-out test split. The temperature experiment uses the first 30 sentences of that evaluation subset.

### Preprocessing

- Read the English and Kannada files as aligned sentence pairs.
- Sample a manageable subset of the parallel corpus.
- Split the data into training, validation, and test sets.
- Tokenize inputs according to each model's requirements.
- Evaluate predictions against the corresponding Kannada references.

The dataset may contain noisy or imperfect translations, which can affect automatic evaluation scores.

## Models and Methodology

### 1. Basic Transformer

A character-level encoder–decoder Transformer implemented and trained from scratch.

- Character-level vocabulary.
- Two encoder layers and two decoder layers.
- Model dimension: 128.
- Attention heads: 4.
- Feed-forward dimension: 512.
- Dropout: 0.1.
- Optimizer: Adam.
- Learning rate: 0.0003.
- Trained for up to five epochs on a smaller training subset.

This model provides a lightweight baseline without pretrained language-model weights.

### 2. mT5-small

A pretrained multilingual text-to-text Transformer fine-tuned for English-to-Kannada translation.

- Architecture: encoder–decoder Transformer.
- Fine-tuning dataset: sampled parallel corpus.
- Training epochs: 2.
- Evaluation: held-out translation examples.
- Validation loss decreased from approximately 6.937 to 6.637.

This experiment investigates whether multilingual pretraining improves translation compared with training a small Transformer from scratch.

### 3. TinyLlama-1.1B

TinyLlama is evaluated as a decoder-only language model using a translation instruction prompt.

- Model: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- Adaptation: prompt-only, without fine-tuning.
- Decoding: greedy generation for the main comparison.
- Maximum generated tokens: 40.

This experiment tests how well a general-purpose instruction-tuned language model performs on English-to-Kannada translation without task-specific training.

### 4. Gemma-2-2B-IT

Gemma is evaluated using an instruction prompt requesting a Kannada translation.

- Model: `google/gemma-2-2b-it`
- Adaptation: prompt-only, without fine-tuning.
- Decoding: greedy generation for the main comparison.
- Maximum generated tokens: 40.

The experiment examines whether a larger instruction-tuned model provides better reference similarity than TinyLlama on the selected evaluation set.

## Evaluation Metrics

**BLEU** measures n-gram overlap between generated translations and reference translations.

**chrF** measures character-level n-gram similarity and is useful for languages where inflection and spelling variation matter.

**Training time** records the training or fine-tuning duration where applicable.

**Inference time per sentence** measures the average time taken to generate a translation under the recorded experimental setup.

**Parameter count** indicates the number of model parameters, not the amount of GPU memory required during inference.

BLEU and chrF are reference-based similarity metrics. They do not independently measure semantic correctness, fluency, or cultural appropriateness.

## Main Results

The following results were recorded from the 200-sentence evaluation set.

| Model | Parameters | Training time (min) | BLEU | chrF | Inference time (sec/sentence) |
|---|---:|---:|---:|---:|---:|
| Basic Transformer | 1,016,127 | 0.301 | 0.04 | 7.11 | 0.4388 |
| mT5-small | 300,176,768 | 33.28 | 0.57 | 12.75 | 0.3242 |
| TinyLlama-1.1B | 1,100,048,384 | N/A | 0.05 | 2.61 | 1.6275 |
| Gemma-2-2B-IT | 2,614,341,888 | N/A | 1.09 | 15.84 | 1.8572 |

*Training time is not applicable to TinyLlama and Gemma because they were evaluated without fine-tuning. Timings depend on the runtime environment and should be interpreted as measurements from this experiment.*

### Observations

- **Gemma** achieved the highest recorded BLEU and chrF scores in the main comparison.
- **mT5-small** performed better than the Basic Transformer and TinyLlama on both reported similarity metrics.
- **Basic Transformer** was substantially smaller than the pretrained models, but its translation quality was limited.
- **TinyLlama** had low reference similarity despite its larger parameter count than mT5's baseline implementation.
- Model size alone did not determine translation quality. Pretraining, tokenization, task adaptation, and decoding behaviour also mattered.

These results should be interpreted cautiously because the models use different training and adaptation strategies, and the evaluation subset is relatively small.

## Temperature Analysis

Temperature controls the randomness of token generation. The experiment compares temperatures of 0.0, 0.7, and 1.0 for TinyLlama and Gemma on the same 30 held-out examples.

| Model | Temperature | BLEU | chrF |
|---|---:|---:|---:|
| TinyLlama | 0.0 | 0.21 | 3.09 |
| TinyLlama | 0.7 | 0.38 | 2.85 |
| TinyLlama | 1.0 | 0.21 | 1.76 |
| Gemma | 0.0 | 1.01 | 17.01 |
| Gemma | 0.7 | 0.84 | 16.76 |
| Gemma | 1.0 | 0.81 | 14.47 |

### Findings

- Gemma achieved its highest BLEU and chrF at temperature 0.0.
- TinyLlama achieved its highest BLEU at temperature 0.7, but its highest chrF at temperature 0.0.
- For both models, temperature 1.0 reduced chrF relative to temperature 0.0.
- Higher randomness did not consistently improve similarity to the reference translations.

The temperature results are preliminary because they use only 30 examples. They should not be assumed to generalise to the entire test set.

## Qualitative Error Analysis

Automatic scores are complemented by inspecting generated translations alongside the English source and Kannada reference.

Potential error types include:

- **Omission:** important details from the source are missing.
- **Hallucination:** the translation introduces unsupported information.
- **Lexical errors:** words are translated incorrectly or inappropriately.
- **Tokenization errors:** rare words, names, and Kannada inflections are mishandled.
- **Cultural nuance loss:** idioms, tone, or context are not preserved.
- **Fluency issues:** generated Kannada may be repetitive, malformed, or unnatural.

A translation that appears fluent is not necessarily faithful to the original meaning. Manual review by Kannada speakers would provide a stronger assessment of adequacy and fluency.

## Responsible AI Considerations

### Bias and language representation

The dataset may not represent every Kannada dialect, regional expression, or domain equally. Translation quality may therefore vary across language varieties and topics.

**Mitigation:** Audit dataset coverage and evaluate examples from diverse domains and language varieties.

### Hallucination and meaning preservation

Models may omit details, alter meaning, or produce plausible but unsupported text.

**Mitigation:** Check names, numbers, negation, instructions, and source fidelity. Require human verification for consequential translations.

### Cultural and linguistic nuance

Literal translations may lose honorifics, idioms, humour, tone, or culturally specific meaning.

**Mitigation:** Include bilingual human assessment and review culturally sensitive examples.

### Sensitive and high-impact content

Errors in medical, legal, financial, emergency, or public-service translations can have serious consequences.

**Mitigation:** Do not use these experimental models as the sole source for high-stakes translations. Use qualified human reviewers.

### Privacy and data handling

Future applications may receive personal or confidential text. External inference services can introduce additional data-handling risks.

**Mitigation:** Minimise personal information and review model licences, usage terms, and privacy requirements before deployment.

### Evaluation limitations

BLEU and chrF measure reference similarity rather than complete semantic correctness. The main comparison uses 200 examples, and the temperature experiment uses 30. Results should not be treated as proof of general translation reliability.

## Repository Structure

The repository is intended to contain the notebook and generated evaluation artifacts.

```text
english-kannada-machine-translation/
├── README.md
├── ia3fgai_NOTEBOOK.ipynb
└── OUTPUTS/
    ├── four_model_comparison.csv
    ├── gemma_metrics.csv
    ├── gemma_predictions.csv
    ├── tinyllama_metrics.csv
    ├── tinyllama_predictions.csv
    ├── qualitative_examples.csv
    ├── responsible_ai_risks.csv
    ├── temperature_analysis.csv
    ├── temperature_sample_outputs.csv
    ├── bleu_comparison.png
    ├── chrf_comparison.png
    ├── inference_time_comparison.png
    ├── parameter_comparison.png
    ├── temperature_bleu.png
    └── temperature_chrf.png
```

The actual output directory should be checked before committing. Include any available prediction files for mT5 and the Basic Transformer if they were saved successfully.

## How to Run

The recommended environment is Kaggle Notebooks.

1. Create or open a Kaggle notebook.
2. Enable GPU acceleration.
3. Enable internet access if required for model downloads.
4. Add the English-to-Kannada dataset through Kaggle's input panel.
5. Run the notebook cells in order.
6. For Gemma, accept the applicable Hugging Face model terms and configure a read token through Kaggle Secrets if authentication is required.
7. Inspect the generated metrics and prediction files.
8. Download the final notebook and outputs from Kaggle and upload them to this repository.

### Dependencies

The notebook uses Python packages including:

- PyTorch
- Transformers
- Datasets
- SentencePiece
- SacreBLEU
- Accelerate
- Pandas
- Matplotlib
- scikit-learn

Package versions and GPU availability can affect reproducibility. Use the installation cell in the notebook and avoid changing the environment midway through a run unless necessary.

## Output Files

The `assignment3_outputs/` directory contains the evaluation results, visualizations, and analysis artifacts generated by the Kaggle notebook.

| File | Description |
|---|---|
| `four_model_comparison.csv` | Consolidated comparison of all four models, including parameter count, training time, BLEU, chrF, and inference time. |
| `gemma_metrics.csv` | Evaluation metrics for Gemma-2-2B-IT using prompt-only translation. |
| `gemma_predictions.csv` | English inputs, Gemma-generated Kannada translations, and reference translations. |
| `tinyllama_metrics.csv` | Evaluation metrics for TinyLlama-1.1B using prompt-only translation. |
| `tinyllama_predictions.csv` | English inputs, TinyLlama-generated Kannada translations, and reference translations. |
| `bleu_comparison.png` | Bar chart comparing BLEU scores across models. |
| `chrf_comparison.png` | Bar chart comparing chrF scores across models. |
| `inference_time_comparison.png` | Comparison of average inference time per sentence. |
| `parameter_comparison.png` | Comparison of model parameter counts using a logarithmic scale. |
| `temperature_analysis.csv` | BLEU, chrF, and inference-time results for TinyLlama and Gemma at temperatures 0.0, 0.7, and 1.0. |
| `temperature_bleu.png` | Plot showing how generation temperature affects BLEU scores. |
| `temperature_chrf.png` | Plot showing how generation temperature affects chrF scores. |
| `temperature_sample_outputs.csv` | Sample translations generated at different temperatures, alongside the reference translations. |
| `qualitative_examples.csv` | Sample translations collected for qualitative comparison and error analysis. |
| `responsible_ai_risks.csv` | Identified translation risks, potential impacts, and proposed mitigations. |

The comparison and temperature-analysis files support quantitative evaluation, while the prediction CSVs enable inspection of individual translations and translation errors.

**Note:** The consolidated comparison uses 200 evaluation examples per model. The temperature analysis uses a smaller subset of 30 examples. Training-time values are not applicable to the prompt-only TinyLlama and Gemma experiments.

## Limitations and Future Work

- Evaluate on a larger, independently curated test set.
- Improve dataset cleaning and remove noisy or misaligned pairs.
- Compare more Kannada-specific and multilingual pretrained models.
- Tune hyperparameters and extend training where computationally feasible.
- Conduct bilingual human evaluation for adequacy, fluency, and cultural appropriateness.
- Investigate terminology preservation, named entities, and rare-word handling.
- Measure uncertainty and add review mechanisms for low-confidence translations.

## Conclusion

This project compares a Transformer trained from scratch, a fine-tuned multilingual encoder–decoder, and two prompt-only decoder-only language models for English-to-Kannada translation.

On the recorded 200-example evaluation set, Gemma achieved the highest BLEU and chrF scores, while mT5 provided a useful pretrained encoder–decoder baseline. The experiments also show that model size alone does not guarantee translation quality and that decoding temperature affects models differently.

The results are exploratory rather than evidence of deployment-ready translation. Larger evaluations, human review, improved data quality, and responsible handling of sensitive content are necessary before practical use.

## Acknowledgements

- Dataset: [English to Kannada Machine Translation Dataset on Kaggle](https://www.kaggle.com/datasets/parvmodi/english-to-kannada-machine-translation-dataset)
- Model authors and maintainers: the respective Hugging Face model repositories for mT5, TinyLlama, and Gemma.

Please follow the dataset's terms and each model's licence and usage requirements.
