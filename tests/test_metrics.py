import numpy as np

from glyphcnn.metrics import accuracy, confusion_matrix, per_class_metrics


def test_accuracy_perfect_and_zero():
    y = np.array([0, 1, 2, 3])
    assert accuracy(y, y) == 1.0
    assert accuracy(y, np.array([9, 9, 9, 9])) == 0.0
    assert accuracy(y, np.array([0, 1, 9, 9])) == 0.5  # 2 of 4 correct


def test_confusion_matrix_shape_and_counts():
    y_true = np.array([0, 0, 1, 2])
    y_pred = np.array([0, 1, 1, 2])
    cm = confusion_matrix(y_true, y_pred, num_classes=3)
    assert cm.shape == (3, 3)
    assert cm[0, 0] == 1 and cm[0, 1] == 1
    assert cm[1, 1] == 1 and cm[2, 2] == 1
    assert cm.sum() == len(y_true)


def test_per_class_metrics_perfect():
    cm = np.diag([3, 2, 4])
    m = per_class_metrics(cm)
    assert m["macro_f1"] == 1.0
    assert m["support"] == [3, 2, 4]


def test_per_class_metrics_handles_empty_class():
    # class 2 never appears -> no division-by-zero, metrics are 0
    cm = np.array([[2, 0, 0], [0, 3, 0], [0, 0, 0]])
    m = per_class_metrics(cm)
    assert m["recall"][2] == 0.0
    assert not np.isnan(m["macro_f1"])
