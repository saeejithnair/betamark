from betamark import ocr


def test_eval():
    def placeholder(x):
        return 0

    ocr.eval(user_func=placeholder)
