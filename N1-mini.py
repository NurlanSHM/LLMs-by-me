import tkinter as tk
import json
import os
import random

class SimpleChatbot:
    def __init__(self, db_path="chatbot_data.json"):
        self.db_path = db_path
        self.memory = {}
        if os.path.exists(self.db_path):
            self.load()

    def load(self):
        with open(self.db_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                for k, v in data.items():
                    key = k.lower()
                    if isinstance(v, list):
                        self.memory[key] = v
                    else:
                        self.memory[key] = [v]
            elif isinstance(data, list):
                for item in data:
                    key = item["input"].lower()
                    resp = item["response"]
                    self.memory.setdefault(key, []).append(resp)

    def save(self):
        with open(self.db_path, "w", encoding="utf-8") as f:
            json.dump(self.memory, f, ensure_ascii=False, indent=2)

    def train(self, user_input, response):
        key = user_input.lower()
        self.memory.setdefault(key, []).append(response)
        self.save()

    def bulk_train(self, training_data):
        for user_input, responses in training_data.items():
            key = user_input.lower()
            if isinstance(responses, list):
                self.memory.setdefault(key, []).extend(responses)
            else:
                self.memory.setdefault(key, []).append(responses)
        self.save()

    def reply(self, user_input):
        ui = user_input.lower()
        if ui in self.memory:
            return random.choice(self.memory[ui])
        best, score = None, 0
        for ki in self.memory:
            s = self.similarity(ui, ki)
            if s > score:
                score, best = s, ki
        if score > 0.5:
            return random.choice(self.memory[best])
        return "Can you teach me how to respond to that? I'm still learning."

    @staticmethod
    def similarity(a, b):
        aw, bw = set(a.split()), set(b.split())
        if not aw or not bw:
            return 0
        return len(aw & bw) / len(aw | bw)

class ChatbotGUI:
    def __init__(self, root, bot):
        self.bot = bot
        self.dark = False
        root.title("NChat")
        root.minsize(500, 400)
        root.configure(bg="#eaeaea")
        root.resizable(True, True)

        self.top_frame = tk.Frame(root, bg="#eaeaea")
        self.top_frame.pack(fill=tk.X, padx=12, pady=(12, 0))

        self.clear_btn = tk.Button(
            self.top_frame, text="Clear Chat", command=self.clear_chat,
            font=("Segoe UI", 9), bg="#d3d3d3", fg="#000", relief="flat", padx=6, pady=2,
            bd=0, highlightthickness=0, cursor="hand2"
        )
        self.clear_btn.pack(side=tk.LEFT)

        self.dark_btn = tk.Button(
            self.top_frame, text="Dark Mode", command=self.toggle_dark,
            font=("Segoe UI", 9), bg="#d3d3d3", fg="#000", relief="flat", padx=6, pady=2,
            bd=0, highlightthickness=0, cursor="hand2"
        )
        self.dark_btn.pack(side=tk.LEFT, padx=(8, 0))

        self.chat_area = tk.Text(
            root, wrap=tk.WORD, state='disabled', font=("Segoe UI", 11),
            bg="#fff", fg="#333", bd=0, padx=10, pady=10, highlightthickness=0
        )
        self.chat_area.pack(padx=12, pady=(12, 0), expand=True, fill=tk.BOTH)

        self.input_frame = tk.Frame(root, bg="#f1f1f1")
        self.input_frame.pack(fill=tk.X, padx=12, pady=12)

        self.input_field = tk.Entry(
            self.input_frame, font=("Segoe UI", 11), bg="#fff", fg="#000",
            insertbackground="#000", relief="flat", bd=0, highlightthickness=0
        )
        self.input_field.pack(side=tk.LEFT, fill=tk.X, expand=True, ipadx=5, ipady=5, padx=(0, 12))
        self.input_field.bind("<Return>", self.send_message)
        self.input_field.focus()

        self.send_btn = tk.Button(
            self.input_frame, text="Send", command=self.send_message,
            font=("Segoe UI", 10, "bold"), bg="#fff", fg="#000",
            activebackground="#e0e0e0", activeforeground="#000",
            relief="flat", bd=0, highlightthickness=0, padx=16, pady=4, cursor="hand2"
        )
        self.send_btn.pack(side=tk.RIGHT)

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

        root = self.chat_area.master
        root.configure(bg=bg)
        self.top_frame.configure(bg=bg)
        self.clear_btn.configure(bg=top_btn_bg, fg=fg)
        self.dark_btn.configure(bg=top_btn_bg, fg=fg,
                                 text="Light Mode" if self.dark else "Dark Mode")
        self.chat_area.configure(bg=chat_bg, fg=fg)
        self.input_frame.configure(bg=input_bg)
        self.input_field.configure(bg=input_bg, fg=fg)
        self.send_btn.configure(bg=input_bg, fg=fg)

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
    # Example bulk training call:
    # bot.bulk_train({"hello": ["Hi!", "Hey!"], "bye": "Goodbye!"})
    root = tk.Tk()
    ChatbotGUI(root, bot)
    root.mainloop()
