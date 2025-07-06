import tkinter as tk
import os
import json
import re
import torch
import random
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from torch.utils.data import Dataset

# Normalize text
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()

class QADataset(Dataset):
    """
    Dataset for fine-tuning: each example is user->assistant concatenated.
    """
    def __init__(self, examples, tokenizer, max_length=512):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        user, assistant = self.examples[idx]
        # Build input: user + eos + assistant + eos
        text = user + self.tokenizer.eos_token + assistant + self.tokenizer.eos_token
        tokens = self.tokenizer(text, truncation=True, max_length=self.max_length, return_tensors="pt")
        tokens["labels"] = tokens.input_ids.clone()
        return {k: v.squeeze() for k,v in tokens.items()}

class TransformerChatbot:
    def __init__(self,
                 model_name: str = "microsoft/DialoGPT-medium",
                 device: str = None):
        # Setup device
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading model {model_name} on {self.device}…")
        # Load model & tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        # Context management
        self.system_prompt = (
            "The following is a conversation between a user and an AI assistant. "
            "The assistant is helpful and informative." + self.tokenizer.eos_token
        )
        self.max_tokens = 1024
        self.reset_context()

    def reset_context(self):
        # start each convo with system prompt
        self.chat_history_ids = self.tokenizer(self.system_prompt, return_tensors="pt").input_ids.to(self.device)

    def reply(self, user_input: str) -> str:
        # Encode and append user input
        user_ids = self.tokenizer(user_input + self.tokenizer.eos_token, return_tensors="pt").input_ids.to(self.device)
        self.chat_history_ids = torch.cat([self.chat_history_ids, user_ids], dim=-1)
        # Truncate history
        if self.chat_history_ids.size(-1) > self.max_tokens:
            self.chat_history_ids = self.chat_history_ids[:, -self.max_tokens:]
        # Attention mask
        attention_mask = torch.ones(self.chat_history_ids.shape, dtype=torch.long).to(self.device)
        # Generate
        try:
            output = self.model.generate(
                self.chat_history_ids,
                attention_mask=attention_mask,
                max_length=self.chat_history_ids.size(-1) + 100,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                top_p=0.9,
                temperature=0.7,
                no_repeat_ngram_size=3,
                num_return_sequences=1
            )
        except Exception as e:
            print("Generation error:", e)
            self.reset_context()
            return "Sorry, something went wrong."
        # Decode
        new_tokens = output[:, self.chat_history_ids.size(-1):]
        response = self.tokenizer.decode(new_tokens[0], skip_special_tokens=True).strip()
        if not response:
            self.reset_context()
            return "Sorry, I didn't catch that."
        # Update history
        self.chat_history_ids = torch.cat([self.chat_history_ids, new_tokens], dim=-1)
        return response

    def train_model(self, qa_json_path: str, output_dir: str = "./fine_tuned_model", epochs: int = 1, batch_size: int = 2):
        """
        Fine-tune the transformer on a JSON file of Q&A pairs: [{"input":..., "response":...}, ...]
        Saves the model to `output_dir`.
        """
        # Load data
        with open(qa_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        examples = []
        for item in data:
            inp = item.get('input', '').strip()
            resp = item.get('response', '').strip()
            if inp and resp:
                examples.append((inp, resp))
        if not examples:
            print("No valid Q&A pairs found in", qa_json_path)
            return
        # Prepare dataset
        dataset = QADataset(examples, self.tokenizer)
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)
        # Training arguments
        args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            save_steps=1000,
            save_total_limit=1,
            logging_steps=100,
        )
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=args,
            data_collator=data_collator,
            train_dataset=dataset,
        )
        trainer.train()
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"Model fine-tuned and saved to {output_dir}")

class PlaceholderEntry(tk.Entry):
    def __init__(self, master=None, placeholder="Type here...", color='grey', **kwargs):
        super().__init__(master, **kwargs)
        self.placeholder = placeholder
        self.placeholder_color = color
        self.default_fg = self['fg']
        self.bind("<FocusIn>", self._clear)
        self.bind("<FocusOut>", self._add)
        self._add()
    def _clear(self, _):
        if self.get() == self.placeholder and self['fg'] == self.placeholder_color:
            self.delete(0, tk.END)
            self['fg'] = self.default_fg
    def _add(self, _=None):
        if not self.get():
            self.insert(0, self.placeholder)
            self['fg'] = self.placeholder_color
    def is_placeholder(self):
        return self.get() == self.placeholder and self['fg'] == self.placeholder_color

class ChatbotGUI:
    def __init__(self, root, bot: TransformerChatbot):
        self.bot = bot
        self.dark = True
        root.title("Trainable AI Chatbot")
        root.geometry("600x450")
        root.configure(bg="#2e2e2e")
        
        # Buttons frame
        ctrl = tk.Frame(root, bg="#2e2e2e")
        ctrl.pack(fill=tk.X, padx=12, pady=(12,0))
        self.clear_btn = tk.Button(ctrl, text="Clear Chat", command=self.clear_chat,
                                   font=("Segoe UI",9), bg="#666", fg="#ddd",
                                   relief="flat", padx=6, pady=2)
        self.clear_btn.pack(side=tk.LEFT)
        self.dark_btn = tk.Button(ctrl, text="Light Mode", command=self.toggle_dark,
                                  font=("Segoe UI",9), bg="#666", fg="#ddd",
                                  relief="flat", padx=6, pady=2)
        self.dark_btn.pack(side=tk.LEFT, padx=8)
        self.train_btn = tk.Button(ctrl, text="Train Model", command=self.launch_training,
                                   font=("Segoe UI",9), bg="#007acc", fg="#fff",
                                   relief="flat", padx=6, pady=2)
        self.train_btn.pack(side=tk.RIGHT)

        # Chat area
        self.chat_area = tk.Text(root, wrap=tk.WORD, state='disabled',
                                 font=("Segoe UI",11), bg="#3c3c3c", fg="#ddd",
                                 bd=0, padx=10, pady=10)
        self.chat_area.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)

        # Input area
        inp = tk.Frame(root, bg="#4a4a4a")
        inp.pack(fill=tk.X, padx=12, pady=(0,12))
        self.input_field = PlaceholderEntry(inp, placeholder="Ask me anything...",
                                            font=("Segoe UI",11), bg="#4a4a4a", fg="#ddd",
                                            insertbackground="#ddd", relief="flat", bd=0)
        self.input_field.pack(side=tk.LEFT, fill=tk.X, expand=True, ipadx=5, ipady=6, padx=(0,12))
        self.input_field.bind("<Return>", self.send_message)
        self.send_btn = tk.Button(inp, text="Send", command=self.send_message,
                                  font=("Segoe UI",10,"bold"), bg="#4a4a4a", fg="#ddd",
                                  relief="flat", bd=0, padx=16, pady=6)
        self.send_btn.pack(side=tk.RIGHT)

    def launch_training(self):
        # Prompt for JSON path (basic)
        path = tk.filedialog.askopenfilename(title="Select Q&A JSON", filetypes=[("JSON files","*.json")])
        if path:
            self._append("Training started...\n")
            self.bot.train_model(path)
            self._append("Training finished. Model saved.\n")

    # Clear chat area
    def clear_chat(self):
        self.chat_area.config(state='normal')
        self.chat_area.delete('1.0', tk.END)
        self.chat_area.config(state='disabled')
        self.bot.reset_context()

    # Toggle dark mode
    def toggle_dark(self):
        self.dark = not self.dark
        bg, fg = ("#2e2e2e", "#ddd") if self.dark else ("#eaeaea", "#333")
        chat_bg = "#3c3c3c" if self.dark else "#fff"
        inp_bg, btn_bg = ("#4a4a4a", "#666") if self.dark else ("#fff", "#d3d3d3")
        self.chat_area.configure(bg=chat_bg, fg=fg)
        self.input_field.configure(bg=inp_bg, fg=fg, insertbackground=fg)
        self.clear_btn.configure(bg=btn_bg, fg=fg)
        self.dark_btn.configure(bg=btn_bg, fg=fg, text="Light Mode" if self.dark else "Dark Mode")
        root.configure(bg=bg)

    def send_message(self, event=None):
        txt = self.input_field.get().strip()
        if not txt or self.input_field.is_placeholder(): return
        self._append(f"You: {txt}\n")
        resp = self.bot.reply(txt)
        self._append(f"AI: {resp}\n\n")
        self.input_field.delete(0, tk.END)

    def _append(self, text):
        self.chat_area.config(state='normal')
        self.chat_area.insert(tk.END, text)
        self.chat_area.see(tk.END)
        self.chat_area.config(state='disabled')

if __name__ == "__main__":
    from tkinter import filedialog
    bot = TransformerChatbot()
    root = tk.Tk()
    app = ChatbotGUI(root, bot)
    root.mainloop()