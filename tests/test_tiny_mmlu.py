from betamark import tiny_mmlu


def test_eval():
    def placeholder(x):
        return "A"

    return tiny_mmlu.run_eval(user_func=placeholder)


if __name__ == "__main__":
    val = test_eval()
    print(val)
