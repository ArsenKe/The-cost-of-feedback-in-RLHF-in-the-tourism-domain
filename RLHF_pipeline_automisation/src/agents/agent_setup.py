from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
from langchain_community.llms import HuggingFacePipeline
import torch

def create_agent():
    # Load base model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("ArsenKe/MT5_large_finetuned_chatbot")
    base_model = AutoModelForSeq2SeqLM.from_pretrained("ArsenKe/MT5_large_finetuned_chatbot")

    # Load LoRA adapter on top of base model
    model = PeftModel.from_pretrained(base_model, "ArsenKe/MT5_DPO_finetuned")

    # Use CPU 
    model.to("cpu")

    llm = HuggingFacePipeline(pipeline=model)

    # Initialize tools
    from ..tools.tourism_tools import TourismTools
    tools = TourismTools()

    return tools


# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# from peft        import PeftModel

# BASE    = "ArsenKe/MT5_large_finetuned_chatbot"
# ADAPTER = "ArsenKe/MT5_DPO_finetuned"

# tok   = AutoTokenizer.from_pretrained(BASE)
# base  = AutoModelForSeq2SeqLM.from_pretrained(BASE)
# model = PeftModel.from_pretrained(base, ADAPTER)

# def chat(prompt: str) -> str:
#     inputs = tok(prompt, return_tensors="pt").to(model.device)
#     out    = model.generate(**inputs, max_new_tokens=100)
#     return tok.decode(out[0], skip_special_tokens=True)
