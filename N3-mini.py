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
            "what's up": ["sup", "whats up", "waddup", "what's going on", "how's it going"],
            "bye": ["goodbye", "see ya", "later", "farewell", "ciao"],
            "thank you": ["thanks", "thx", "ty", "thank you"],
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

        for key, alias_list in self.aliases.items():
            for alias in alias_list:
                pattern = r'\b' + re.escape(alias) + r'\b'
                if re.search(pattern, ui):
                    if key in self.memory:
                        if self.debug:
                            print(f"[Alias match] input='{ui}' matched alias '{alias}' -> key '{key}'")
                        return random.choice(self.memory[key])

        if ui in self.memory:
            if self.debug:
                print(f"[Exact match] input='{ui}'")
            return random.choice(self.memory[ui])

        keys = list(self.memory.keys())
        if keys:
            self._update_vectorizer()
            doc_vectors = self.vectorizer.transform(keys)
            ui_vector = self.vectorizer.transform([ui])
            sims = cosine_similarity(ui_vector, doc_vectors).flatten()
            best_idx = sims.argmax()
            best_score = sims[best_idx]

            if self.debug:
                print(f"[TF-IDF similarity] input='{ui}'")
                for i, key in enumerate(keys):
                    print(f"  similarity to '{key}': {sims[i]:.3f}")

            threshold = 0.25 if len(ui.split()) <= 3 else 0.3

            if best_score >= threshold:
                best_key = keys[best_idx]
                if self.debug:
                    print(f"[TF-IDF reply] best_key='{best_key}', score={best_score:.3f}")
                return random.choice(self.memory[best_key])

        return "Can you teach me how to respond to that? I'm still learning."

    def _update_vectorizer(self):
        if not self._fit_needed:
            return
        docs = list(self.memory.keys())
        if docs:
            self.vectorizer.fit(docs)
        self._fit_needed = False

class PlaceholderEntry(tk.Entry):
    def __init__(self, master=None, placeholder="PLACEHOLDER", color='grey', **kwargs):
        super().__init__(master, **kwargs)
        self.placeholder = placeholder
        self.placeholder_color = color
        self.default_fg_color = self['fg']

        self.bind("<FocusIn>", self._clear_placeholder)
        self.bind("<FocusOut>", self._add_placeholder)

        self._add_placeholder()

    def _clear_placeholder(self, e):
        if self['fg'] == self.placeholder_color:
            self.delete(0, tk.END)
            self['fg'] = self.default_fg_color

    def _add_placeholder(self, e=None):
        if not self.get():
            self.insert(0, self.placeholder)
            self['fg'] = self.placeholder_color

class ChatbotGUI:
    def __init__(self, root, bot):
        self.bot = bot
        self.dark = True  # Dark mode default
        self.root = root

        self.root.title("NChat")
        self.root.minsize(500, 400)
        self.root.configure(bg="#2e2e2e")
        self.root.resizable(True, True)

        # Top buttons frame
        self.top_frame = tk.Frame(self.root, bg="#2e2e2e")
        self.top_frame.pack(fill=tk.X, padx=12, pady=(12, 0))

        self.clear_btn = tk.Button(
            self.top_frame, text="Clear Chat", command=self.clear_chat,
            font=("Segoe UI", 9), bg="#666", fg="#ddd", relief="flat",
            padx=6, pady=2, bd=0, highlightthickness=0, cursor="hand2"
        )
        self.clear_btn.pack(side=tk.LEFT)

        self.dark_btn = tk.Button(
            self.top_frame, text="Light Mode", command=self.toggle_dark,
            font=("Segoe UI", 9), bg="#666", fg="#ddd", relief="flat",
            padx=6, pady=2, bd=0, highlightthickness=0, cursor="hand2"
        )
        self.dark_btn.pack(side=tk.LEFT, padx=(8, 0))

        self.train_toggle_btn = tk.Button(
            self.top_frame, text="Train", command=self.toggle_train_panel,
            font=("Segoe UI", 9), bg="#666", fg="#ddd", relief="flat",
            padx=6, pady=2, bd=0, highlightthickness=0, cursor="hand2"
        )
        self.train_toggle_btn.pack(side=tk.LEFT, padx=(8, 0))

        # Chat display
        self.chat_area = tk.Text(
            self.root, wrap=tk.WORD, state='disabled', font=("Segoe UI", 11),
            bg="#3c3c3c", fg="#ddd", bd=0, padx=10, pady=10, highlightthickness=0
        )
        self.chat_area.pack(padx=12, pady=(12, 0), expand=True, fill=tk.BOTH)

        # Input frame
        self.input_frame = tk.Frame(self.root, bg="#4a4a4a")
        self.input_frame.pack(fill=tk.X, padx=12, pady=12)

        self.input_field = PlaceholderEntry(
            self.input_frame, placeholder="Write here...",
            font=("Segoe UI", 11), bg="#4a4a4a", fg="#ddd",
            insertbackground="#ddd", relief="flat", bd=0, highlightthickness=0
        )
        self.input_field.pack(side=tk.LEFT, fill=tk.X, expand=True, ipadx=5, ipady=5, padx=(0, 12))
        self.input_field.bind("<Return>", self.send_message)
        self.input_field.focus()

        self.send_btn = tk.Button(
            self.input_frame, text="Send", command=self.send_message,
            font=("Segoe UI", 10, "bold"), bg="#4a4a4a", fg="#ddd",
            activebackground="#5c5c5c", activeforeground="#fff",
            relief="flat", bd=0, highlightthickness=0, padx=16, pady=4, cursor="hand2"
        )
        self.send_btn.pack(side=tk.RIGHT)

        # Training panel (hidden by default)
        self.train_panel = tk.Frame(self.root, bg="#3c3c3c", bd=1, relief="solid")
        # Position top-right, will be packed/unpacked dynamically
        # We'll pack it to the top_frame side right with some padding later.

        # Prompt input
        tk.Label(self.train_panel, text="Prompt:", bg="#3c3c3c", fg="#ddd", font=("Segoe UI", 9)).grid(row=0, column=0, sticky="w", padx=6, pady=(6, 2))
        self.prompt_entry = tk.Entry(self.train_panel, font=("Segoe UI", 10), bg="#4a4a4a", fg="#ddd", insertbackground="#ddd", relief="flat")
        self.prompt_entry.grid(row=1, column=0, padx=6, pady=(0, 8), sticky="ew")

        # Answer input
        tk.Label(self.train_panel, text="Answer:", bg="#3c3c3c", fg="#ddd", font=("Segoe UI", 9)).grid(row=2, column=0, sticky="w", padx=6, pady=(0, 2))
        self.answer_entry = tk.Entry(self.train_panel, font=("Segoe UI", 10), bg="#4a4a4a", fg="#ddd", insertbackground="#ddd", relief="flat")
        self.answer_entry.grid(row=3, column=0, padx=6, pady=(0, 8), sticky="ew")

        # Train button inside panel
        self.train_btn = tk.Button(
            self.train_panel, text="Train", command=self.train_from_panel,
            font=("Segoe UI", 10, "bold"), bg="#4a4a4a", fg="#ddd",
            activebackground="#5c5c5c", activeforeground="#fff",
            relief="flat", bd=0, highlightthickness=0, padx=12, pady=4, cursor="hand2"
        )
        self.train_btn.grid(row=4, column=0, pady=(0, 6))

        self.train_panel.grid_columnconfigure(0, weight=1)

        self.train_panel_visible = False

    def toggle_train_panel(self):
        if self.train_panel_visible:
            self.train_panel.pack_forget()
            self.train_panel_visible = False
        else:
            # Place top-right below top_frame with some padding
            self.train_panel.pack(side=tk.TOP, anchor="ne", padx=12, pady=6)
            self.train_panel_visible = True

    def train_from_panel(self):
        prompt = self.prompt_entry.get().strip()
        answer = self.answer_entry.get().strip()
        if not prompt or not answer:
            self._append("nChat: Please fill both Prompt and Answer to train.\n\n")
            return
        self.bot.train(prompt, answer)
        self._append(f"nChat trained: '{prompt}' → '{answer}'\n\n")
        self.prompt_entry.delete(0, tk.END)
        self.answer_entry.delete(0, tk.END)

    def clear_chat(self):
        self.chat_area.config(state='normal')
        self.chat_area.delete(1.0, tk.END)
        self.chat_area.config(state='disabled')

    def toggle_dark(self):
        self.dark = not self.dark
        bg = "#2e2e2e" if self.dark else "#eaeaea"
        fg = "#ddd" if self.dark else "#333"
        chat_bg = "#3c3c3c" if self.dark else "#fff"
        input_bg = "#4a4a4a" if self.dark else "#fff"
        top_btn_bg = "#666" if self.dark else "#d3d3d3"

        self.root.configure(bg=bg)
        self.top_frame.configure(bg=bg)
        self.clear_btn.configure(bg=top_btn_bg, fg=fg)
        self.dark_btn.configure(bg=top_btn_bg, fg=fg,
                                 text="Light Mode" if self.dark else "Dark Mode")
        self.train_toggle_btn.configure(bg=top_btn_bg, fg=fg)
        self.chat_area.configure(bg=chat_bg, fg=fg)
        self.input_frame.configure(bg=input_bg)
        self.input_field.configure(bg=input_bg, fg=fg, insertbackground=fg)
        self.send_btn.configure(bg=input_bg, fg=fg)
        self.train_panel.configure(bg=chat_bg)
        # Update train panel labels and entries colors
        for child in self.train_panel.winfo_children():
            if isinstance(child, tk.Label):
                child.configure(bg=chat_bg, fg=fg)
            elif isinstance(child, tk.Entry) or isinstance(child, tk.Button):
                child.configure(bg=input_bg, fg=fg, insertbackground=fg)

    def send_message(self, event=None):
        text = self.input_field.get().strip()
        if not text:
            return
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
    bot = SimpleChatbot(debug=False)
    if os.path.exists("mass_training.json"):
        with open("mass_training.json", "r", encoding="utf-8") as f:
            data = json.load(f)
            bot.bulk_train(data)
            print("✅ Bot trained from mass_training.json")

    root = tk.Tk()
    app = ChatbotGUI(root, bot)
    root.mainloop()
