from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import numpy as np
import matplotlib.pyplot as plt

#1. Load and Prepare the Dataset
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#2. Train an SVM with Linear and RBF Kernel
svm_linear = SVC(kernel='linear')
svm_linear.fit(X_train, y_train)
svm_rbf = SVC(kernel='rbf')
svm_rbf.fit(X_train, y_train)
y_pred_linear = svm_linear.predict(X_test)
y_pred_rbf = svm_rbf.predict(X_test)
print("Linear SVM Accuracy:", accuracy_score(y_test, y_pred_linear))
print("RBF SVM Accuracy:", accuracy_score(y_test, y_pred_rbf))

#3. Visualize Decision Boundary (for 2D data)
X_vis = X_train[:, :2]  
y_vis = y_train
svm_vis = SVC(kernel='linear')
svm_vis.fit(X_vis, y_vis)
xx, yy = np.meshgrid(np.linspace(X_vis[:, 0].min(), X_vis[:, 0].max(), 100),
                     np.linspace(X_vis[:, 1].min(), X_vis[:, 1].max(), 100))
Z = svm_vis.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(X_vis[:, 0], X_vis[:, 1], c=y_vis, edgecolor='k')
plt.title("Linear SVM Decision Boundary")
plt.show()

#4. Tune Hyperparameters (C and Gamma)
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [0.01, 0.1, 1, 10]}
grid_search = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=5)
grid_search.fit(X_train, y_train)
print("Best parameters:", grid_search.best_params_)

#5. Evaluate Performance with Cross-Validation
cv_scores = cross_val_score(SVC(kernel='rbf', C=grid_search.best_params_['C'],
                                gamma=grid_search.best_params_['gamma']),
                            X_train, y_train, cv=5)
print("Cross-Validation Accuracy:", cv_scores.mean())
