import yaml
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

with open('config.YAML') as file:
    config = yaml.safe_load(file)


def knn(x_train, y_train):
    """Train a K-Nearest Neighbour classifier on the training data provided

    :param x_train:Independent variables
    :param y_train:Dependent variable
    :return:Trained KNN classifier
    """
    classifier = KNeighborsClassifier(n_neighbors=config['knn_n_neighbors'],
                                      weights=config['knn_weights'],
                                      algorithm=config['knn_algorithm'],
                                      leaf_size=config['knn_leaf_size'],
                                      p=config['knn_p'],
                                      metric=config['knn_metric'],
                                      metric_params=config['knn_metric_params'],
                                      n_jobs=config['knn_n_jobs']
                                      )

    return classifier.fit(x_train, y_train)


def balanced_bagging(x_train, y_train):
    """Train a Balanced Bagging classifier with a Decision Tree base estimator on the training data provided

    :param x_train:Independent variables
    :param y_train:Dependent variable
    :return:Trained Balanced Bagging classifier
    """
    classifier = BalancedBaggingClassifier(base_estimator=DecisionTreeClassifier(),
                                           n_estimators=config['bb_n_estimators'],
                                           max_samples=config['bb_max_samples'],
                                           max_features=config['bb_max_features'],
                                           bootstrap=config['bb_bootstrap'],
                                           bootstrap_features=config['bb_bootstrap_features'],
                                           oob_score=config['bb_oob_score'],
                                           warm_start=config['bb_warm_start'],
                                           sampling_strategy=config['bb_sampling_strategy'],
                                           replacement=config['bb_replacement'],
                                           n_jobs=config['bb_n_jobs'],
                                           random_state=config['bb_random_state'],
                                           verbose=config['bb_verbose'],
                                           sampler=config['bb_sampler']
                                           )

    classifier.fit(x_train, y_train)

    return classifier


def random_forest(x_train, y_train):
    classifier = RandomForestClassifier(n_estimators=config['rf_n_estimators'],
                                        criterion=config['rf_criterion'],
                                        max_depth=config['rf_max_depth'],
                                        min_samples_split=config['rf_min_samples_split'],
                                        min_samples_leaf=config['rf_min_samples_leaf'],
                                        min_weight_fraction_leaf=config['rf_min_weight_fraction_leaf'],
                                        max_features=config['rf_max_features'],
                                        max_leaf_nodes=config['rf_max_leaf_nodes'],
                                        min_impurity_decrease=config['rf_min_impurity_decrease'],
                                        bootstrap=config['rf_bootstrap'],
                                        oob_score=config['rf_oob_score'],
                                        n_jobs=config['rf_n_jobs'],
                                        random_state=config['rf_random_state'],
                                        verbose=config['rf_verbose'],
                                        warm_start=config['rf_warm_start'],
                                        class_weight=config['rf_class_weight'],
                                        ccp_alpha=config['rf_ccp_alpha'],
                                        max_samples=config['rf_max_samples']
                                        )
    classifier.fit(x_train, y_train)

    return classifier
