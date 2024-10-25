from betamark import ocr


def test_eval():
    def placeholder(x):
        return 0

    ocr.run_eval(user_func=placeholder)
