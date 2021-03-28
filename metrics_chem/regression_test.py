import numpy as np
import regression as r

y_true_1 = np.array([3, -0.5, 2, 7])
y_pred_1 = np.array([2.5, 0.0, 2, 8])

y_true_2 = np.array([[0.5, 1], [-1, 1], [7, -6]])
y_pred_2 = np.array([[0, 2], [-1, 2], [8, -5]])

print(r.mean_absolute_percentage_error(y_true_1, y_pred_1))