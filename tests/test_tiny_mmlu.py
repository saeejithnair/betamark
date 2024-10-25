from betamark import tiny_mmlu


def test_eval():
    def placeholder(x):
        return "A"

    tiny_mmlu.run_eval(user_func=placeholder)
