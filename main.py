from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


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
    print("Accuracy:", round(accuracy_score(y_test, predictions) * 100, 2), "%")
    print(classification_report(y_test, predictions, target_names=iris.target_names))

    sample = [[5.1, 3.5, 1.4, 0.2]]
    result = model.predict(sample)[0]
    print("Example flower prediction:", iris.target_names[result])


if __name__ == "__main__":
    main()
