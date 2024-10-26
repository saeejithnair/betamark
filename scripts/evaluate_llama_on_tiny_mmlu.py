import torch
import wandb
from transformers import AutoTokenizer, AutoModelForCausalLM
from betamark import tiny_mmlu
import random
import datasets
import argparse

# Load LLAMA model and tokenizer
MODEL_NAME = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")

# Load the dataset
ds = datasets.load_dataset("tinyBenchmarks/tinyMMLU")

def llama_predict(prompt, temperature):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=1, temperature=temperature, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response[-1]

def run_evaluation(config=None):
    with wandb.init(config=config):
        config = wandb.config
        
        def predict_wrapper(prompt):
            return llama_predict(prompt, config.temperature)
        
        print(f"Evaluating LLAMA model on Tiny MMLU dataset (Split: {config.split}, Seed: {config.seed}, Temperature: {config.temperature})...")
        
        # Set random seed
        random.seed(config.seed)
        torch.manual_seed(config.seed)
        
        results = tiny_mmlu.run_eval(user_func=predict_wrapper, split=config.split)
        
        # Log overall accuracy
        wandb.log({"accuracy": results["acc"]})
        
        # Create a table for detailed results
        table = wandb.Table(columns=["Index", "Question", "Choices","Ground Truth", "Prediction", "Correct"])
        ANSWER_MAPPING = {
            "0": "A",
            "1": "B",
            "2": "C",
            "3": "D"
        }
        # Log individual question results
        for i, item in enumerate(ds[config.split]):
            input_formatted = item["input_formatted"]
            ground_truth = str(item["answer"])
            prediction = str(predict_wrapper(input_formatted))
            question = item["question"]
            choices = item["choices"]
            # Join choices into a string with prefix A, B, C, D
            choices_str = ""
            for i, choice in enumerate(choices):
                choices_str += f"{ANSWER_MAPPING[str(i)]}. {choice}\n"

            prediction_stripped = tiny_mmlu.strip_model_resp(prediction)
            ground_truth_stripped = tiny_mmlu.strip_model_resp(ground_truth)
            
            is_correct = prediction_stripped == ANSWER_MAPPING[ground_truth_stripped]
            
            table.add_data(f"Question {i+1}", question, choices_str, ANSWER_MAPPING[ground_truth_stripped], prediction_stripped, is_correct)

        # Log the table
        wandb.log({"detailed_results": table})

        print(f"Evaluation results: {results}")

# Define the sweep configuration
def get_sweep_config(split):
    return {
        'method': 'grid',
        'name': f'llama-tiny-mmlu-sweep-{split}',
        'metric': {'goal': 'maximize', 'name': 'accuracy'},
        'parameters': 
        {
            'seed': {'values': list(range(10))},  # 10 different seeds
            'temperature': {'values': [0.05, 0.1, 0.2, 0.5, 1.0]},  # 5 temperature values
            'split': {'value': split}
        }
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLAMA evaluation on Tiny MMLU dataset")
    parser.add_argument('--split', type=str, choices=['dev', 'test'], default='dev',
                        help='Dataset split to use for evaluation')
    args = parser.parse_args()

    sweep_configuration = get_sweep_config(args.split)
    sweep_id = wandb.sweep(sweep=sweep_configuration, project="llama-tiny-mmlu-evaluation")
    wandb.agent(sweep_id, function=run_evaluation, count=50)  # 10 seeds * 5 temperatures = 50 runs
