# LLMs-by-me

A collection of AI chatbots I built while learning, progressing from simple
keyword lookups to a fine-tuned DialoGPT model.

## The progression

| File | Approach | Notes |
|---|---|---|
| `N1-mini.py` | Keyword / JSON lookup | Rule-based `SimpleChatbot` + Tkinter GUI |
| `N1-violet.py` | Keyword / JSON lookup | Variant of N1 |
| `N2.py` | TF-IDF + cosine similarity | Smarter matching via scikit-learn |
| `N3-mini.py` | Transformer inference | DialoGPT text generation |
| `N3-o.py` | Transformer inference | Variant of N3 |
| `N4-max.py` | Fine-tuning | Trains DialoGPT with Hugging Face `Trainer` |
| `chatbot_data.json` | Data | The training/conversation data |

## Requirements

- Python 3.8+
- `torch` + `transformers` (for N3 / N4)
- `scikit-learn` (for N2)

## Usage

Each script is self-contained. Run any of them directly:

```bash
python N2.py
```

For N3/N4, set your model and data paths inside the script before running.
