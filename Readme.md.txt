# Thiomi Multilingual Machine Translation (MMT): Kikuyu ⇄ English

👉 **[Live Demo on Hugging Face Spaces] https://huggingface.co/spaces/NjeriKahoro/kikuyu-english-translator ** 👉 **[Model Checkpoint on Hugging Face Hub] https://huggingface.co/NjeriKahoro/nllb-200-Thiomi-Kik-Eng-3 **

An end-to-end Machine Translation (MT) pipeline featuring a fine-tuned, bidirectional...

# Thiomi Multilingual Machine Translation (MMT): Kikuyu ⇄ English

An end-to-end Machine Translation (MT) pipeline featuring a fine-tuned, bidirectional **NLLB-200 (No Language Left Behind)** model optimized for high-quality translation between **English (eng_Latn)** and **Gĩkũyũ / Kikuyu (kik_Latn)**. This repository also contains an interactive deployment interface powered by **Gradio**.

---

##  Project Overview

Low-resource language technology often relies on massive datasets or brittle parallel architectures. This project adapts Meta's `nllb-200-distilled-600M` via targeted sequence-to-sequence fine-tuning on localized language pairs. 

### Key Features
* **True Bidirectional Performance:** A single model checkpoint handling both `English ➔ Kikuyu` and `Kikuyu ➔ English` without code changes, using exact target prefix token controls.
* **Balanced Shuffling:** The text preprocessing architecture flattens parallel rows into explicit language directions and interleaves them perfectly ($50\%$ train/test per path direction) to prevent model steering or catastrophic forgetting.
* **Optimized VRAM Profile:** Implemented using 8-bit Paged AdamW optimizers, gradient checkpointing, and expandable segment allocation configs, enabling stable training loops on single-GPU instances.

---

## 📊 Dataset & Preprocessing

The model is fine-tuned using the parallel text repository: [`NjeriKahoro/thiomi-multilingual-text`](https://huggingface.co/datasets/NjeriKahoro/thiomi-multilingual-text).

* **Train Samples:** 10,906 total rows (5,453 base parallel pairs flattened bidirectionally)
* **Test Samples:** 2,730 total rows (1,365 base parallel pairs flattened bidirectionally)

### Parallel Flattening Strategy
To optimize multilingual parameter updates, raw text columns are converted dynamically into specific translation directions prior to tokenization:

| Source Code | Input String (Tokenized) | Target Code | Output String (Labels) |
| :--- | :--- | :--- | :--- |
| `kik_Latn` | Mbura nene na riũwa inini. | `eng_Latn` | Heavy rain and a little sun. |
| `eng_Latn` | It's a good day today. | `kik_Latn` | Rũciũ rũrĩa nĩ rũega. |

---

## 🛠️ Training Configuration

The model was optimized using the Hugging Face `Seq2SeqTrainer` framework with the following architectural hyperparameters:

* **Base Model:** `facebook/nllb-200-distilled-600M`
* **Optimizer:** `paged_adamw_8bit`
* **Learning Rate:** $2 \times 10^{-5}$ (with a linear warmup ratio of $0.1$ across 5 epochs)
* **Batch Sizing:** Effective Batch Size = $16$ (`per_device_train_batch_size=1` with `gradient_accumulation_steps=16`)
* **Precision:** Floating-point half precision (`fp16=True`)
* **Evaluation Metric:** Character n-gram F-score (`chrf`) optimized with checkpoint saving tracking the highest-scoring epoch.

---

## 📈 Performance & Curves

The training logs show steady convergence of translation perplexity over the epochs:

| Epoch | Training Loss | Validation Loss | ChrF Score | BLEU Score |
| :--- | :--- | :--- | :--- | :--- |
| 1 | 54.5629 | 1.5838 | 45.5783 | 19.5862 |
| 2 | 50.6309 | 1.5361 | 46.6380 | 20.2887 |

*Visual loss plots and translation metrics are saved automatically inside the repo folder as `loss_curves.png` and `generation_metrics.png`.*

---

## 🖥️ Interactive Web Interface (Gradio App)

The deployment script `app.py` runs an interactive web interface allowing users to seamlessly transition between translation targets.

```bash
pip install gradio transformers torch
python app.py