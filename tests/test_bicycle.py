from betamark import bicycle


def test_eval():
    def placeholder(x):
        return 0

    bicycle.run_eval(user_func=placeholder)
