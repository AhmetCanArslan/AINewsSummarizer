import os
import threading
import queue
import torch
import tkinter as tk
from tkinter import scrolledtext, ttk
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_DIR = "./t5_finetuned"
FALLBACK_MODEL = "Turkish-NLP/t5-efficient-small-MLSUM-TR-fine-tuned"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

model_source = MODEL_DIR if os.path.isdir(MODEL_DIR) else FALLBACK_MODEL
print(f"Loading model from: {model_source}")

tokenizer = AutoTokenizer.from_pretrained(model_source)
model = AutoModelForSeq2SeqLM.from_pretrained(model_source).to(device)
model.eval()

result_q = queue.Queue()

def generate_in_thread(input_text, max_length, num_beams, do_sample):
    try:
        prefix = "" 
        inp = prefix + input_text.strip()
        inputs = tokenizer(
            inp,
            return_tensors="pt",
            truncation=True,
            padding="longest",
            max_length=512
        ).to(device)

        with torch.no_grad():
            gen_ids = model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams,
                do_sample=do_sample,
                early_stopping=True,
                no_repeat_ngram_size=3
            )
        summary = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
    except Exception as e:
        summary = f"[HATA] {e}"
    result_q.put(summary)

# UI
class ChatDemo(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("T5 Chat Demo")
        self.geometry("800x600")

        self.chat = scrolledtext.ScrolledText(self, wrap=tk.WORD, state="disabled", font=("Arial", 11))
        self.chat.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        ctl_frame = tk.Frame(self)
        ctl_frame.pack(fill=tk.X, padx=8, pady=(0,8))

        self.entry = tk.Text(ctl_frame, height=4, wrap=tk.WORD)
        self.entry.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        right_frame = tk.Frame(ctl_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(8,0))

        self.gen_btn = tk.Button(right_frame, text="Gönder", width=10, command=self.on_send)
        self.gen_btn.pack(pady=(0,8))

        self.maxlen_var = tk.IntVar(value=64)
        tk.Label(right_frame, text="Max len").pack()
        tk.Spinbox(right_frame, from_=16, to=256, textvariable=self.maxlen_var, width=6).pack()

        self.beams_var = tk.IntVar(value=4)
        tk.Label(right_frame, text="Beams").pack()
        tk.Spinbox(right_frame, from_=1, to=8, textvariable=self.beams_var, width=6).pack()

        self.sample_var = tk.BooleanVar(value=False)
        tk.Checkbutton(right_frame, text="Sample", variable=self.sample_var).pack(pady=(6,0))

        self.after(200, self.check_result_queue)

    def append_chat(self, role, text):
        self.chat.configure(state="normal")
        if role == "user":
            self.chat.insert(tk.END, f"YOU: {text}\n\n")
        else:
            self.chat.insert(tk.END, f"MODEL: {text}\n\n")
        self.chat.see(tk.END)
        self.chat.configure(state="disabled")

    def on_send(self):
        user_text = self.entry.get("1.0", tk.END).strip()
        if not user_text:
            return
        self.entry.delete("1.0", tk.END)
        self.append_chat("user", user_text)
        self.append_chat("system", "Üretiliyor...")

        # disable button while generating
        self.gen_btn.configure(state="disabled")

        t = threading.Thread(
            target=generate_in_thread,
            args=(user_text, self.maxlen_var.get(), self.beams_var.get(), self.sample_var.get()),
            daemon=True
        )
        t.start()

    def check_result_queue(self):
        try:
            while True:
                summary = result_q.get_nowait()
                self.append_chat("model", summary)
                self.gen_btn.configure(state="normal")
        except queue.Empty:
            pass
        self.after(200, self.check_result_queue)

if __name__ == "__main__":
    app = ChatDemo()
    app.mainloop()