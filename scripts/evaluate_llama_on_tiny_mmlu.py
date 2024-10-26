import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from betamark import tiny_mmlu

# Load LLAMA model and tokenizer
MODEL_NAME = "meta-llama/Llama-3.2-1B"  # You may need to adjust this based on the specific LLAMA model you want to use
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")

def llama_predict(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=1, temperature=0.2, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract the last character (which should be the model's answer)
    return response[-1]

def main():
    print("Evaluating LLAMA model on Tiny MMLU dataset...")
    results = tiny_mmlu.run_eval(user_func=llama_predict)
    print(f"Evaluation results: {results}")

if __name__ == "__main__":
    main()
