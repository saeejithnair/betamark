import datasets
import tqdm

VERBOSE = False

ds = datasets.load_dataset("katarinagresova/Genomic_Benchmarks_human_ocr_ensembl")

train_subsample = ds["train"].shuffle(seed=42)[0:200]

if VERBOSE:
    print(train_subsample)


def run_eval(user_func) -> dict:
    correct_count = 0
    total_count = 0
    for i in tqdm.trange(len(train_subsample)):
        model_input = train_subsample["seq"][i]
        ground_truth = train_subsample["label"][i]
        model_prediction = user_func(model_input)
        total_count += 1
        if ground_truth == model_prediction:
            correct_count += 1
    acc = correct_count / total_count
    return {"acc": acc}
