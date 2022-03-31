from imblearn.ensemble import BalancedBaggingClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier


class Classifiers:
    def __init__(self, df):
        '''

        :param df:
        '''
    def knn(x_train, y_train, x_test):
        classifier = KNeighborsClassifier(n_neighbors=10)

        classifier.fit(x_train, y_train)

        preds = classifier.predict(x_test)

        return preds

    def balanced_bagging(x_train, y_train, x_test):
        classifier = BalancedBaggingClassifier(base_estimator=DecisionTreeClassifier(),
                                               sampling_strategy='not majority',
                                               replacement=False,
                                               random_state=42)
        classifier.fit(x_train, y_train)

        preds = classifier.predict(x_test)

        return preds

    def random_forest(x_train, y_train, x_test):
        classifier = RandomForestClassifier(random_state=0)
        classifier.fit(x_train, y_train)

        preds = classifier.predict(x_test)

        return preds
