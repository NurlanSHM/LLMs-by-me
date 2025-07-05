# Transformer Chatbot (DialoGPT)

A simple yet trainable AI chatbot based on Microsoft’s DialoGPT model using Hugging Face Transformers and PyTorch.

---

## Features

- Conversational AI with context memory  
- Trainable on your custom data  
- Supports GPU acceleration (CUDA) if available  
- Simple Tkinter GUI included  

---

## Requirements

- Python 3.7+  
- [PyTorch](https://pytorch.org/get-started/locally/) (with CUDA if you want GPU support)  
- Transformers library  
- Datasets library (Hugging Face)  

---

## Installation

Install dependencies with pip:

pip install torch transformers datasets


If you want GPU support, install the correct PyTorch version for your CUDA version by following instructions from [PyTorch official](https://pytorch.org/get-started/locally/).

---

## Usage

### Run chatbot GUI

python chatbot.py



This launches a simple Tkinter GUI. Type your messages and get AI responses.

### Training

You can train the chatbot on your own conversational data in JSON format (see example below).

Run training script:

python train.py --data_path your_data.json --model_name microsoft/DialoGPT-medium


---

## Data Format

The training data should be a JSON file with a list of conversations:

[
  {
    "input": "Hello, how are you?",
    "response": "I'm doing great, thank you!"
  },
  {
    "input": "Tell me a joke.",
    "response": "Why did the chicken cross the road? To get to the other side!"
  }
]


Notes
Training large models requires significant GPU memory (8GB+ recommended).

Keep an eye on max sequence length to avoid memory errors.

For best results, fine-tune with a batch size of 2-4 and adjust learning rate accordingly.

Troubleshooting
If you get CUDA errors, check if your GPU and PyTorch versions match.

Blank or repetitive responses? Try resetting context or adjusting generation parameters like temperature and top_p.

License
