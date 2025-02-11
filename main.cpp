import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.datasets import make_classification

# Generate synthetic dataset
X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]  # Probability scores for positive class

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
TP, FN, FP, TN = cm[1, 1], cm[1, 0], cm[0, 1], cm[0, 0]

# Performance Metrics
TPR = TP / (TP + FN)  # Sensitivity / Recall
FPR = FP / (FP + TN)
TNR = TN / (TN + FP)  # Specificity
FNR = FN / (FN + TP)
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)

print("Confusion Matrix:")
print(cm)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print(f"TPR (Sensitivity / Recall): {TPR:.2f}")
print(f"FPR: {FPR:.2f}")
print(f"TNR (Specificity): {TNR:.2f}")
print(f"FNR: {FNR:.2f}")
print(f"Precision: {Precision:.2f}")
print(f"Recall: {Recall:.2f}")

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Random classifier line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-AUC Curve')
plt.legend()
plt.show()
