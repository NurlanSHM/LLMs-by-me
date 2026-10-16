# LLMs-by-me

A collection of chatbots built while learning, from exact phrase lookup to transformer-based text generation.

## The progression

| File | Approach | Notes |
|---|---|---|
| `N1-mini.py` | Exact phrase lookup | JSON-backed chatbot with a Tkinter GUI |
| `N1-violet.py` | TF-IDF + cosine similarity | Adds approximate matching and aliases |
| `N2.py` | TF-IDF + cosine similarity | Expanded alias matching and debug output |
| `N3-mini.py` | TF-IDF + cosine similarity | Later chatbot variant; loads `mass_training.json` when present |
| `N3-o.py` | TF-IDF + cosine similarity | Expanded aliases; loads `mass_training.json` when present |
| `N4-max.py` | DialoGPT text generation and fine-tuning | Uses Hugging Face Transformers; choose training data in the GUI |
| `chatbot_data.json` | Data | Chat and training examples for the JSON-backed versions |

## Requirements

- Python 3
- Tkinter for the desktop interfaces (included with many Python installations)
- `scikit-learn` for `N1-violet.py`, `N2.py`, `N3-mini.py`, and `N3-o.py`
- `torch` and `transformers` for `N4-max.py`

Install the machine-learning packages you need for the script you plan to run. PyTorch installation options vary by operating system and hardware; see the [PyTorch install guide](https://pytorch.org/get-started/locally/).

## Usage

Run any script from the repository directory. For example:

```bash
python N2.py
```

N4 uses `microsoft/DialoGPT-medium` by default. You can pass another Hugging Face model ID or a local model directory:

```bash
python N4-max.py --model microsoft/DialoGPT-medium
```

Click **Train Model** to select a JSON file containing Q&A objects with `input` and `response` fields. N4 saves the fine-tuned model and tokenizer to `./fine_tuned_model`; run it later with:

```bash
python N4-max.py --model ./fine_tuned_model
```
