from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)


# =========================================================
# EVALUATE MODEL
# =========================================================

def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(y_test, predictions)

    recall = recall_score(y_test, predictions)

    f1 = f1_score(y_test, predictions)

    roc_auc = roc_auc_score(y_test, probabilities)

    print("\n=================================================")
    print("CHURN MODEL EVALUATION")
    print("=================================================\n")

    print(f"Accuracy  : {accuracy:.4f}")

    print(f"Precision : {precision:.4f}")

    print(f"Recall    : {recall:.4f}")

    print(f"F1 Score  : {f1:.4f}")

    print(f"ROC AUC   : {roc_auc:.4f}")

    print("\n=================================================")
    print("CLASSIFICATION REPORT")
    print("=================================================\n")

    print(classification_report(y_test, predictions))

    print("\n=================================================")
    print("CONFUSION MATRIX")
    print("=================================================\n")

    print(confusion_matrix(y_test, predictions))