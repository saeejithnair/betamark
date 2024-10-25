import datasets
import tqdm

VERBOSE = False

ds = datasets.load_dataset("katarinagresova/Genomic_Benchmarks_human_ocr_ensembl")

test_subsample = ds["test"].shuffle(seed=42)[0:200]

if VERBOSE:
    print(test_subsample)


def eval(user_func) -> dict:
    correct_count = 0
    total_count = 0
    for i in tqdm.trange(len(test_subsample)):
        model_input = test_subsample["seq"][i]
        ground_truth = test_subsample["label"][i]
        model_prediction = user_func(model_input)
        total_count += 1
        if ground_truth == model_prediction:
            correct_count += 1
    acc = correct_count / total_count
    return {"acc": acc}
