import tkinter as tk
import json
import os
import random
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def clean_text(text):
    return re.sub(r"[^\w\s']", '', text.lower()).strip()

class SimpleChatbot:
    def __init__(self, db_path="chatbot_data.json", debug=False):
        self.db_path = db_path
        self.memory = {}
        self.debug = debug
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
        self._fit_needed = False
        self.aliases = {
            "hello": ["hi", "hey", "howdy", "hiya", "yo"],
            "what's up": ["sup", "whats up", "waddup", "how's it going"],
            "bye": ["goodbye", "see ya", "later", "farewell", "ciao"],
            "thank you": ["thanks", "thx", "ty", "thank you"]
        }
        if os.path.exists(self.db_path):
            self.load()
        self._update_vectorizer()

    def load(self):
        with open(self.db_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                for k, v in data.items():
                    key = clean_text(k)
                    self.memory[key] = v if isinstance(v, list) else [v]
            elif isinstance(data, list):
                for item in data:
                    key = clean_text(item.get("input", ""))
                    resp = item.get("response", "")
                    if key:
                        self.memory.setdefault(key, []).append(resp)
        self._fit_needed = True

    def save(self):
        with open(self.db_path, "w", encoding="utf-8") as f:
            json.dump(self.memory, f, ensure_ascii=False, indent=2)

    def train(self, user_input, response):
        key = clean_text(user_input)
        self.memory.setdefault(key, []).append(response)
        self.save()
        self._fit_needed = True
        self._update_vectorizer()

    def bulk_train(self, training_data):
        for user_input, responses in training_data.items():
            key = clean_text(user_input)
            if isinstance(responses, list):
                self.memory.setdefault(key, []).extend(responses)
            else:
                self.memory.setdefault(key, []).append(responses)
        self.save()
        self._fit_needed = True
        self._update_vectorizer()

    def reply(self, user_input):
        ui = clean_text(user_input)
        # 1. Alias match
        for key, alias_list in self.aliases.items():
            for alias in alias_list:
                if re.search(r'\b' + re.escape(alias) + r'\b', ui):
                    return random.choice(self.memory.get(key, ["..."]))
        # 2. Exact
        if ui in self.memory:
            return random.choice(self.memory[ui])
        # 3. Semantic
        keys = list(self.memory.keys())
        if keys:
            self._update_vectorizer()
            doc_vectors = self.vectorizer.transform(keys)
            ui_vector = self.vectorizer.transform([ui])
            sims = cosine_similarity(ui_vector, doc_vectors).flatten()
            best_idx = sims.argmax()
            best_score = sims[best_idx]
            threshold = 0.25 if len(ui.split()) <= 3 else 0.3
            if best_score >= threshold:
                return random.choice(self.memory[keys[best_idx]])
        # 4. Fallback
        return "Can you teach me how to respond to that? I'm still learning."

    def _update_vectorizer(self):
        if not self._fit_needed:
            return
        docs = list(self.memory.keys())
        if docs:
            self.vectorizer.fit(docs)
        self._fit_needed = False

class ChatbotGUI:
    def __init__(self, root, bot):
        self.bot = bot
        self.dark = True
        self.root = root

        root.title("NChat")
        root.minsize(500, 400)
        root.configure(bg="#2e2e2e")
        root.resizable(True, True)

        # Top buttons
        self.top_frame = tk.Frame(root, bg="#2e2e2e")
        self.top_frame.pack(fill=tk.X, padx=12, pady=(12, 0))

        self.clear_btn = tk.Button(
            self.top_frame, text="Clear Chat", command=self.clear_chat,
            font=("Segoe UI", 9), bg="#666", fg="#ddd", relief="flat",
            padx=10, pady=4, bd=0, cursor="hand2"
        )
        self.clear_btn.pack(side=tk.LEFT, padx=(0,8))

        self.dark_btn = tk.Button(
            self.top_frame, text="Light Mode", command=self.toggle_dark,
            font=("Segoe UI", 9), bg="#666", fg="#ddd", relief="flat",
            padx=10, pady=4, bd=0, cursor="hand2"
        )
        self.dark_btn.pack(side=tk.LEFT)

        # Chat area
        self.chat_area = tk.Text(
            root, wrap=tk.WORD, state='disabled', font=("Segoe UI", 11),
            bg="#3c3c3c", fg="#ddd", bd=0, padx=12, pady=12
        )
        self.chat_area.pack(padx=12, pady=(12, 0), expand=True, fill=tk.BOTH)

        # Input row
        self.input_frame = tk.Frame(root, bg="#3c3c3c")
        self.input_frame.pack(fill=tk.X, padx=12, pady=12)

        self.input_field = tk.Entry(
            self.input_frame, font=("Segoe UI", 11), bg="#4a4a4a", fg="#ddd",
            insertbackground="#ddd", relief="flat", bd=0
        )
        self.input_field.pack(side=tk.LEFT, fill=tk.X, expand=True,
                              ipadx=8, ipady=6, padx=(0,10))
        self.input_field.bind("<Return>", self.send_message)
        self.input_field.focus()

        self.send_btn = tk.Button(
            self.input_frame, text="↵", command=self.send_message,
            font=("Segoe UI", 12), bg="#4a4a4a", fg="#ddd", relief="flat",
            padx=12, pady=6, bd=0, cursor="hand2"
        )
        self.send_btn.pack(side=tk.RIGHT)

    def clear_chat(self):
        self.chat_area.config(state='normal')
        self.chat_area.delete(1.0, tk.END)
        self.chat_area.config(state='disabled')

    def toggle_dark(self):
        self.dark = not self.dark
        if self.dark:
            bg, fg = "#2e2e2e", "#ddd"
            in_bg, in_fg = "#4a4a4a", "#ddd"
            btn_bg = "#666"
            self.dark_btn.config(text="Light Mode")
        else:
            bg, fg = "#eaeaea", "#000"
            in_bg, in_fg = "#fff", "#000"
            btn_bg = "#d3d3d3"
            self.dark_btn.config(text="Dark Mode")

        # Apply colors
        self.root.configure(bg=bg)
        self.top_frame.configure(bg=bg)
        self.clear_btn.configure(bg=btn_bg, fg=fg)
        self.dark_btn.configure(bg=btn_bg, fg=fg)
        self.chat_area.configure(bg=("#3c3c3c" if self.dark else "#fff"), fg=fg)
        self.input_frame.configure(bg=("#3c3c3c" if self.dark else "#f1f1f1"))
        self.input_field.configure(bg=in_bg, fg=in_fg, insertbackground=in_fg)
        self.send_btn.configure(bg=in_bg, fg=in_fg)

    def send_message(self, event=None):
        text = self.input_field.get().strip()
        if not text:
            return

        if text.lower().startswith('train:'):
            parts = text[6:].split('=', 1)
            if len(parts) == 2:
                q, r = parts[0].strip(), parts[1].strip()
                self.bot.train(q, r)
                self._append(f"nChat trained: '{q}' → '{r}'\n\n")
            else:
                self._append("nChat: Training format is train: question=answer\n\n")
        else:
            self._append(f"You: {text}\n")
            resp = self.bot.reply(text)
            self._append(f"nChat: {resp}\n\n")

        self.input_field.delete(0, tk.END)

    def _append(self, txt):
        self.chat_area.config(state='normal')
        self.chat_area.insert(tk.END, txt)
        self.chat_area.see(tk.END)
        self.chat_area.config(state='disabled')

if __name__ == "__main__":
    bot = SimpleChatbot()
    if os.path.exists("mass_training.json"):
        with open("mass_training.json", "r", encoding="utf-8") as f:
            bot.bulk_train(json.load(f))
    root = tk.Tk()
    ChatbotGUI(root, bot)
    root.mainloop()
