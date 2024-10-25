from betamark import tiny_mmlu


def test_eval():
    def placeholder(x):
        return "A"

    tiny_mmlu.eval(user_func=placeholder)
    print(tiny_mmlu)
    # assert True
