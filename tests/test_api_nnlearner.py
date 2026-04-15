import numpy as np

from base_attentive.api.property import NNLearner


class Child(NNLearner):
    def __init__(self, width=8):
        self.width = width


class Parent(NNLearner):
    def __init__(self, child=None, values=None):
        self.child = child
        self.values = values


def test_repr_nested_learner():
    obj = Parent(child=Child(width=16))
    text = repr(obj)

    assert "Parent(" in text
    assert "Child(" in text
    assert "width=16" in text


def test_repr_array_summary():
    arr = np.zeros((4, 3), dtype=np.float32)
    obj = Parent(values=arr)

    text = repr(obj)

    assert "numpy.ndarray" in text
    assert "shape=(4, 3)" in text
    assert "dtype=float32" in text


def test_repr_depth_limit():
    class Deep(NNLearner):
        _repr_max_depth = 1

        def __init__(self, child=None):
            self.child = child

    obj = Deep(child=Deep(child=Deep()))
    text = repr(obj)

    assert "Deep(" in text
    assert "..." in text or "Deep(...)" in text


def test_repr_cycle_safe():
    class Node(NNLearner):
        def __init__(self, child=None):
            self.child = child

    a = Node()
    a.child = a

    text = repr(a)

    assert "Node(" in text
    assert "..." in text


def test_repr_html_escaped():
    class HtmlLearner(NNLearner):
        def __init__(self, name="<unsafe>"):
            self.name = name

    obj = HtmlLearner()
    html_text = obj._repr_html_()

    assert "&lt;unsafe&gt;" in html_text
    assert "<pre" in html_text


def test_str_readable():
    obj = Parent(
        child=Child(width=12),
        values=[1, 2, 3],
    )
    text = str(obj)

    assert text.startswith("Parent:")
    assert "child:" in text
    assert "values:" in text
