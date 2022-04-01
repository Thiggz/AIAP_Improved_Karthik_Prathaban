from sklearn import metrics


def default_metrics(y_true, y_pred, name='Your_Model', show=1):
    results = {'accuracy': round(metrics.accuracy_score(y_true, y_pred), 3),
               'precision': round(metrics.precision_score(y_true, y_pred), 3),
               'recall': round(metrics.recall_score(y_true, y_pred), 3)}
    if show == 1: print('Model: ' + name + '\n', results, '\n')
    return results


def detailed_metrics(y_true, y_pred, name='Your_Model'):
    results = default_metrics(y_true, y_pred, show=0)

    results_add = {'F1_Score': round(metrics.f1_score(y_true, y_pred), 3),
                   'Specificity': round(metrics.precision_score(y_true, y_pred), 3),
                   'Confusion Matrix': metrics.confusion_matrix(y_true, y_pred)}

    results.update(results_add)
    print('Model: ' + name, '\n')
    for key, value in results.items():
        print(key + ":", '\n', value)
