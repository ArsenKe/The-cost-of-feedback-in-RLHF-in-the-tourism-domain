from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
import gradio as gr

BASE    = "ArsenKe/MT5_large_finetuned_chatbot"
ADAPTER = "ArsenKe/MT5_DPO_finetuned"

tokenizer = AutoTokenizer.from_pretrained(BASE)
base_model = AutoModelForSeq2SeqLM.from_pretrained(BASE)
model = PeftModel.from_pretrained(base_model, ADAPTER)

def chat(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=100)
    return tokenizer.decode(out[0], skip_special_tokens=True)

gr.Interface(chat, "text", "text").launch()


# import torch

# if torch.cuda.is_available():
#     print("CUDA is available!")
#     print(f"CUDA Version: {torch.version.cuda}")
#     print(f"PyTorch Version: {torch.__version__}")
# else:
#     print("CUDA is not available.")