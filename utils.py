from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score, roc_auc_score

def calculate_metrics(preds, labels):
    print(f"Confusion_matrix:" , confusion_matrix(labels, preds))
    print(f"Accuracy: ", accuracy_score(labels, preds))
    print(f"F1_score:", f1_score(labels, preds))

    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)

    print(f"Precison: {precision}")
    print(f"Recall: {recall}")

    roc_auc_score(labels, preds)