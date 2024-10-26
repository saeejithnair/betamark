import datasets
import tqdm

VERBOSE = True

ds = datasets.load_dataset("tinyBenchmarks/tinyMMLU")

if VERBOSE:
    print(ds)
    for split in ['dev', 'test']:
        print(f"{split} split:")
        print(ds[split][0]["input_formatted"])
        print("----------")
        print(ds[split][0]["answer"])
        print("\n")


def strip_model_resp(model_resp: str) -> str:
    if not isinstance(model_resp, str):
        model_resp = str(model_resp)
    model_resp = model_resp.strip()
    model_resp = model_resp[0] if model_resp else ""
    model_resp = model_resp.upper()
    return model_resp


def run_eval(user_func, split='dev') -> dict:
    answer_map = {"A": 0, "B": 1, "C": 2, "D": 3}

    correct_count = 0
    total = 0
    for i in tqdm.trange(len(ds[split])):
        input_formatted = ds[split][i]["input_formatted"]
        ground_truth_answer = ds[split][i]["answer"]
        predicted_answer = answer_map[strip_model_resp(user_func(input_formatted))]
        total += 1
        if ground_truth_answer == predicted_answer:
            correct_count += 1
    acc = correct_count / total
    return {"acc": acc}
