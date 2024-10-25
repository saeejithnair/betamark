# betamark

An experimental benchmark meant for portability and finite compute.

### Installation Instructions

```bash
pip install git+https://github.com/exobyte-labs/betamark.git
```

To update the installation:

```bash
pip install git+https://github.com/exobyte-labs/betamark.git -U
```


For HackOS 1, the current code base uses train or dev datasets. For the final benchmarking near demo time, this repository will be updated to use validation datasets. 

### TinyMMLU Eval
```python
from betamark import tiny_mmlu

def placeholder(x):
    """
    Params:
    -------
    x: string that is a text prompt

    Returns:
    --------
    y_pred: str, ideally a single character {A, B, C, D} corresponding to a multiple choice answer from MMLU or tinyMMLU
    """
    return "A"

tiny_mmlu.run_eval(user_func=placeholder)

```

### OCR Binary Classification 

```python
from betamark import ocr

def placeholder(x):
    """
    Params:
    -------
    x: string representing a genomic sequence

    Returns:
    --------
    y_pred: int where 0 is negative (not an OCR) or 1 (is an OCR)
    """

    return 0

ocr.run_eval(user_func=placeholder)

```

### Bike Detection on MS Coco Subsample

```python
from betamark import bike

def placeholder(x):
    """
    Params:
    -------
    x: NumPy array representation of an image (dimensions are non-fixed)

    Returns:
    --------
    y_pred: int where 0 is negative (no bike) or 1 (there is a bike)
    """
    return 0

bike.run_eval(user_func=placeholder)

```
