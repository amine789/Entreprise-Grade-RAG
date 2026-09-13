from sklearn.metrics import classification_report, confusion_matrix

from enterprise_rag.router import route_query


def evaluate_router(dataset: list[dict]) -> dict:
    """Run route_query over a labeled dataset and score its predictions.

    dataset: list of {"query": str, "label": str} where label is the
    expected route_query action.
    """
    predictions = []
    for item in dataset:
        result = route_query(item["query"])
        predictions.append(
            {
                "query": item["query"],
                "expected": item["label"],
                "predicted": result["action"],
                "reason": result.get("reason", ""),
            }
        )

    y_true = [p["expected"] for p in predictions]
    y_pred = [p["predicted"] for p in predictions]
    labels = sorted(set(y_true) | set(y_pred))

    return {
        "predictions": predictions,
        "labels": labels,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels),
        "report": classification_report(
            y_true, y_pred, labels=labels, output_dict=True, zero_division=0
        ),
    }
