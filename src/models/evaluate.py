from sklearn.metrics import classification_report,confusion_matrix


def evaluate_model(model,X_test,y_test):
    preds = model.predict(X_test)
    report = classification_report(y_test,preds)
    cm = confusion_matrix(y_test,preds)

    print("Classification report:\n",report)
    print("\nConfusion matrix:\n",cm)