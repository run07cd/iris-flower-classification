from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def main():
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data,
        iris.target,
        test_size=0.2,
        random_state=42,
        stratify=iris.target,
    )

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=200),
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    print("Test accuracy:", round(accuracy_score(y_test, predictions) * 100, 2), "%")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, target_names=iris.target_names))

    print("Confusion matrix (rows = actual, columns = predicted):")
    print(confusion_matrix(y_test, predictions))

    scores = cross_val_score(model, iris.data, iris.target, cv=5, scoring="accuracy")
    print("\n5-fold cross-validation accuracy:", [round(score, 3) for score in scores])
    print("Mean cross-validation accuracy:", round(scores.mean() * 100, 2), "%")

    sample = [[5.1, 3.5, 1.4, 0.2]]
    result = model.predict(sample)[0]
    print("\nExample flower prediction:", iris.target_names[result])


if __name__ == "__main__":
    main()
