classification_step = False # else regression_step

import numpy as np
import pandas
import itertools
from sklearn.model_selection import KFold
from sklearn import preprocessing
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.preprocessing import PolynomialFeatures
from sklearn import preprocessing

from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.pipeline import make_pipeline
import gc

keep_incomplete_data_vals = [False, True]

normalize_data_vals = [False, True]
standardize_data_vals = [True]

full_one_hot_vals = [False, True]

# Hyperparameters
polynomial_degree_vals = np.arange(1, 12).tolist()

# Logistic hyperparameters
logistic_penalty_vals = ["l2"]
logistic_C_vals = [1.0, 2.0]
logistic_solver_vals = ['newton-cg', 'sag']
logistic_multiclass_vals = ['multinomial']

# Random forest
random_forest_n_estimators_vals = [50, 100]
random_forest_criterion_vals = ['entropy', 'gini']

# KNN
knn_n_neighbours_vals = np.arange(1, 5).tolist()
knn_algorithm_vals = ['auto'] # , 'ball_tree', 'kd_tree', 'brute'
knn_p_vals = [1, 2]

# Regression general
reg_fit_intercept_vals = [True, False]

# Lasso
lasso_alpha_vals = [1.0]

# Ridge
ridge_alpha_vals = [1.0]
ridge_solver_vals = ['auto'] # , 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'

# Elastic Net
el_net_alpha_vals = [1.0]
el_net_l1_ratio_vals = [0.5]

hyperparameters_cls_vals = [keep_incomplete_data_vals,
normalize_data_vals,
standardize_data_vals,
full_one_hot_vals,
polynomial_degree_vals,
logistic_penalty_vals,
logistic_C_vals,
logistic_solver_vals,
logistic_multiclass_vals,
random_forest_n_estimators_vals,
random_forest_criterion_vals,
knn_n_neighbours_vals,
knn_algorithm_vals,
knn_p_vals]

hyperparameters_reg_vals = [keep_incomplete_data_vals,
normalize_data_vals,
standardize_data_vals,
full_one_hot_vals,
polynomial_degree_vals,
reg_fit_intercept_vals,
lasso_alpha_vals,
ridge_alpha_vals,
ridge_solver_vals,
el_net_alpha_vals,
el_net_l1_ratio_vals]

hyperparameters_combinations_cls = list(itertools.product(*hyperparameters_cls_vals))
hyperparameters_combinations_reg = list(itertools.product(*hyperparameters_reg_vals))

def run_grid():

    data = pandas.read_csv('https://raw.githubusercontent.com/GabiAD/Dogs-Breed-Classification-and-Longevity-Regression/master/dataset.csv')

    data = data.drop(columns=['Owner Name'])

    longevity_norm = np.sqrt(np.sum(data["Longevity(yrs)"] ** 2))
    longevity_std = data["Longevity(yrs)"].std()
    longevity_mean = data["Longevity(yrs)"].mean()


    kf = KFold(5)

    try:
        modified_data
    except NameError:
        modified_data = pandas.DataFrame(columns=data.columns.tolist())

    if not keep_incomplete_data:
        # sunt eliminate datele incomplete
        data = data.dropna()
        # datele ramase sunt amestecate pentru a le alege mai tarziu pe primele 25% pentru validare
        data = data.sample(frac=1).reset_index(drop=True)
        modified_data = pandas.DataFrame(columns=data.columns.tolist())

        k_fold_indices = kf.split(data)
    else:
        if data["Height(cm)"].isnull().sum() > 0 and modified_data.empty:
            incomplete_rows = data["Height(cm)"].isnull()
            incomplete_data = data[incomplete_rows]

            # sunt eliminate datele incomplete
            data = data.dropna()
            # datele ramase sunt amestecate pentru a le alege mai tarziupe primele 25% pentru validare
            data = data.sample(frac=1).reset_index(drop=True)

            k_fold_indices = kf.split(data)

            for row in incomplete_data.iterrows():
                current_row = row[1]
                longevity_std = data["Longevity(yrs)"].std()
                weight_std = data["Weight(g)"].std()

                interest_data = pandas.DataFrame()

                interest_data = data[(data["Breed Name"] == row[1]["Breed Name"]) &
                                       (data["Energy level"] == row[1]["Energy level"]) &
                                       (data["Attention Needs"] == row[1]["Attention Needs"]) &
                                       (data["Coat Lenght"] == row[1]["Coat Lenght"]) &
                                       (data["Sex"] == row[1]["Sex"]) &
                                       (data["Longevity(yrs)"] <= row[1]["Longevity(yrs)"] + longevity_std) &
                                       (data["Longevity(yrs)"] >= row[1]["Longevity(yrs)"] - longevity_std) &
                                       (data["Weight(g)"] <= row[1]["Weight(g)"] + weight_std) &
                                       (data["Weight(g)"] >= row[1]["Weight(g)"] - weight_std)]

                mean_height = interest_data["Height(cm)"].mean()
                current_row["Height(cm)"] = mean_height

                if interest_data["Height(cm)"].count() > 0:
                    modified_data = modified_data.append(pandas.Series(current_row.values.tolist(), index=modified_data.columns.tolist()), ignore_index=True)


    if normalize_data:
        data_length = len(data)
        concatenated_data = data.append(modified_data).reset_index()

        concatenated_data["Weight(g)"] = preprocessing.normalize([concatenated_data["Weight(g)"]]).reshape(-1)
        concatenated_data["Height(cm)"] = preprocessing.normalize([concatenated_data["Height(cm)"]]).reshape(-1)
        concatenated_data["Longevity(yrs)"] = preprocessing.normalize([concatenated_data["Longevity(yrs)"]]).reshape(-1)

        data["Weight(g)"] = concatenated_data["Weight(g)"][:data_length].values
        data["Height(cm)"] = concatenated_data["Height(cm)"][:data_length].values
        data["Longevity(yrs)"] = concatenated_data["Longevity(yrs)"][:data_length].values

        modified_data["Weight(g)"] = concatenated_data["Weight(g)"][data_length:].values
        modified_data["Height(cm)"] = concatenated_data["Height(cm)"][data_length:].values
        modified_data["Longevity(yrs)"] = concatenated_data["Longevity(yrs)"][data_length:].values

    elif standardize_data:
        data_length = len(data)
        concatenated_data = data.append(modified_data).reset_index()

        concatenated_data["Weight(g)"] = (concatenated_data["Weight(g)"] - concatenated_data["Weight(g)"].mean())/concatenated_data["Weight(g)"].std()
        concatenated_data["Height(cm)"] = (concatenated_data["Height(cm)"] - concatenated_data["Height(cm)"].mean())/concatenated_data["Height(cm)"].std()
        concatenated_data["Longevity(yrs)"] = (concatenated_data["Longevity(yrs)"] - concatenated_data["Longevity(yrs)"].mean())/concatenated_data["Longevity(yrs)"].std()

        data["Weight(g)"] = concatenated_data["Weight(g)"][:data_length].values
        data["Height(cm)"] = concatenated_data["Height(cm)"][:data_length].values
        data["Longevity(yrs)"] = concatenated_data["Longevity(yrs)"][:data_length].values

        modified_data["Weight(g)"] = concatenated_data["Weight(g)"][data_length:].values
        modified_data["Height(cm)"] = concatenated_data["Height(cm)"][data_length:].values
        modified_data["Longevity(yrs)"] = concatenated_data["Longevity(yrs)"][data_length:].values

    original_features = data[["Weight(g)", "Height(cm)", "Energy level", "Attention Needs", "Coat Lenght", "Sex"]]
    original_labels_cls = data[["Breed Name"]]
    original_labels_reg = data[["Longevity(yrs)"]]

    modified_features = modified_data[["Weight(g)", "Height(cm)", "Energy level", "Attention Needs", "Coat Lenght", "Sex"]]
    modified_labels_cls = modified_data[["Breed Name"]]
    modified_labels_reg = modified_data[["Longevity(yrs)"]]

    unique_breeds = original_labels_cls["Breed Name"].unique()

    gen = ((i, unique_breeds[i]) for i in range(len(unique_breeds)))
    class_dict = dict(gen)

    if full_one_hot:
        features = pandas.get_dummies(original_features, columns=['Energy level'], prefix = ['Energy level'])
        features = pandas.get_dummies(features, columns=['Attention Needs'], prefix = ['Attention Needs'])
        features = pandas.get_dummies(features, columns=['Coat Lenght'], prefix = ['Coat Lenght'])
        features = pandas.get_dummies(features, columns=['Sex'], prefix = ['Sex'])

        modified_features = pandas.get_dummies(modified_features, columns=['Energy level'], prefix = ['Energy level'])
        modified_features = pandas.get_dummies(modified_features, columns=['Attention Needs'], prefix = ['Attention Needs'])
        modified_features = pandas.get_dummies(modified_features, columns=['Coat Lenght'], prefix = ['Coat Lenght'])
        modified_features = pandas.get_dummies(modified_features, columns=['Sex'], prefix = ['Sex'])
    else:
        features = pandas.get_dummies(original_features, columns=['Sex'], prefix = ['Sex'])

        coat_length_mapper = {0: 'short', 1: 'med', 2: 'long'}
        features['Coat Lenght'] = features['Coat Lenght'].astype('category')
        features['Coat Lenght'] = features['Coat Lenght'].cat.reorder_categories(['short', 'med', 'long'], ordered=True)
        features['Coat Lenght'] = features['Coat Lenght'].cat.codes

        attention_needs_mapper = {0: 'med', 1: 'high'}
        features['Attention Needs'] = features['Attention Needs'].astype('category')
        features['Attention Needs'] = features['Attention Needs'].cat.reorder_categories(['med', 'high'], ordered=True)
        features['Attention Needs'] = features['Attention Needs'].cat.codes

        energy_level_mapper = {0: 'low', 1: 'med', 2: 'high'}
        features['Energy level'] = features['Energy level'].astype('category')
        features['Energy level'] = features['Energy level'].cat.reorder_categories(['low', 'med', 'high'], ordered=True)
        features['Energy level'] = features['Energy level'].cat.codes

        if not modified_features.empty:
            modified_features = pandas.get_dummies(modified_features, columns=['Sex'], prefix = ['Sex'])

            coat_length_mapper = {0: 'short', 1: 'med', 2: 'long'}
            modified_features['Coat Lenght'] = modified_features['Coat Lenght'].astype('category')
            modified_features['Coat Lenght'] = modified_features['Coat Lenght'].cat.reorder_categories(['short', 'med', 'long'], ordered=True)
            modified_features['Coat Lenght'] = modified_features['Coat Lenght'].cat.codes

            attention_needs_mapper = {0: 'med', 1: 'high'}
            modified_features['Attention Needs'] = modified_features['Attention Needs'].astype('category')
            modified_features['Attention Needs'] = modified_features['Attention Needs'].cat.reorder_categories(['med', 'high'], ordered=True)
            modified_features['Attention Needs'] = modified_features['Attention Needs'].cat.codes

            energy_level_mapper = {0: 'low', 1: 'med', 2: 'high'}
            modified_features['Energy level'] = modified_features['Energy level'].astype('category')
            modified_features['Energy level'] = modified_features['Energy level'].cat.reorder_categories(['low', 'med', 'high'], ordered=True)
            modified_features['Energy level'] = modified_features['Energy level'].cat.codes

    # labels_cls = pandas.get_dummies(original_labels_cls, columns=['Breed Name'], prefix = ['Breed Name'])
    labels_cls = original_labels_cls.copy()
    labels_cls = labels_cls.replace(list(class_dict.values()), list(class_dict.keys()))
    labels_reg = original_labels_reg.copy()

    modified_cls = modified_labels_cls.copy()
    modified_labels_cls = modified_labels_cls.replace(list(class_dict.values()), list(class_dict.keys()))
    modified_labels_reg = modified_labels_reg.copy()


    X = features.values.tolist()
    y_cls = labels_cls.values.tolist()
    y_cls = np.reshape(y_cls, (-1))
    y_reg = labels_reg.values.tolist()
    y_reg = np.reshape(y_reg, (-1))

    modified_X = modified_features.values.tolist()
    modified_y_cls = modified_labels_cls.values.tolist()
    modified_y_cls = np.reshape(modified_y_cls, (-1))
    modified_y_reg = modified_labels_reg.values.tolist()
    modified_y_reg = np.reshape(modified_y_reg, (-1))

    if classification_step:

        X = np.array(X)

        X_train, X_test = train_test_split(X, test_size=0.25, random_state=2)
        y_train, y_test = train_test_split(y_cls, test_size=0.25, random_state=2)

        logistic_regression_accuracy = []
        knn_accuracy = []
        random_forest_accuracy = []
        logistic_regression_precision = []
        knn_precision = []
        random_forest_precision = []
        logistic_regression_f1 = []
        knn_f1 = []
        random_forest_f1 = []

        for train_index, val_index in kf.split(X_train):
            X_fold_train, X_fold_val = X_train[train_index], X_train[val_index]
            y_fold_train, y_fold_val = y_train[train_index], y_train[val_index]

            if not modified_features.empty:
                X_fold_train = np.concatenate([X_fold_train, modified_X], axis=0)
                y_fold_train = np.concatenate([y_fold_train, modified_y_cls], axis=0)

            poly_feat = PolynomialFeatures(degree=polynomial_degree)
            X_preprocessed = poly_feat.fit_transform(X_fold_train)

            logistic_regression_model = LogisticRegression(penalty=logistic_penalty, C=logistic_C, multi_class=logistic_multiclass, solver=logistic_solver)
            random_forest_model = RandomForestClassifier(n_estimators=random_forest_n_estimators, criterion=random_forest_criterion)
            knn_model = KNeighborsClassifier(n_neighbors=knn_n_neighbours, p=knn_p, algorithm=knn_algorithm)

            logistic_regression_model.fit(X_preprocessed, y_fold_train)
            knn_model.fit(X_preprocessed, y_fold_train)
            random_forest_model.fit(X_preprocessed, y_fold_train)

            logistic_regression_precision.append(precision_score(y_fold_val, logistic_regression_model.predict(poly_feat.fit_transform(X_fold_val)), average="weighted"))
            knn_precision.append(precision_score(y_fold_val, knn_model.predict(poly_feat.fit_transform(X_fold_val)), average="weighted"))
            random_forest_precision.append(precision_score(y_fold_val, random_forest_model.predict(poly_feat.fit_transform(X_fold_val)), average="weighted"))

            logistic_regression_accuracy.append(accuracy_score(y_fold_val, logistic_regression_model.predict(poly_feat.fit_transform(X_fold_val))))
            knn_accuracy.append(accuracy_score(y_fold_val, knn_model.predict(poly_feat.fit_transform(X_fold_val))))
            random_forest_accuracy.append(accuracy_score(y_fold_val, random_forest_model.predict(poly_feat.fit_transform(X_fold_val))))

            logistic_regression_f1.append(f1_score(y_fold_val, logistic_regression_model.predict(poly_feat.fit_transform(X_fold_val)), average="weighted"))
            knn_f1.append(f1_score(y_fold_val, knn_model.predict(poly_feat.fit_transform(X_fold_val)), average="weighted"))
            random_forest_f1.append(f1_score(y_fold_val, random_forest_model.predict(poly_feat.fit_transform(X_fold_val)), average="weighted"))

    if not classification_step:


        X = np.array(X)

        X_train, X_test = train_test_split(X, test_size=0.25, random_state=2)
        y_train, y_test = train_test_split(y_reg, test_size=0.25, random_state=2)

        linear_regression_mse = []
        lasso_mse = []
        ridge_mse = []
        elastic_net_mse = []

        for train_index, val_index in kf.split(X_train):
            X_fold_train, X_fold_val = X_train[train_index], X_train[val_index]
            y_fold_train, y_fold_val = y_train[train_index], y_train[val_index]

            if not modified_features.empty:
                X_fold_train = np.concatenate([X_fold_train, modified_X], axis=0)
                y_fold_train = np.concatenate([y_fold_train, modified_y_cls], axis=0)

            linear_regression_model = make_pipeline(PolynomialFeatures(degree=polynomial_degree), LinearRegression(fit_intercept=reg_fit_intercept))
            linear_regression_model.fit(X_fold_train, y_fold_train)
            lasso_model = make_pipeline(PolynomialFeatures(degree=polynomial_degree), Lasso(fit_intercept=reg_fit_intercept, alpha=lasso_alpha))
            lasso_model.fit(X_fold_train, y_fold_train)
            ridge_model = make_pipeline(PolynomialFeatures(degree=polynomial_degree), Ridge(fit_intercept=reg_fit_intercept, alpha=ridge_alpha, solver=ridge_solver))
            ridge_model.fit(X_fold_train, y_fold_train)
            elastic_net_model = make_pipeline(PolynomialFeatures(degree=polynomial_degree), ElasticNet(fit_intercept=reg_fit_intercept, alpha=el_net_alpha, l1_ratio=el_net_l1_ratio))
            elastic_net_model.fit(X_fold_train, y_fold_train)

            if normalize_data:
                linear_regression_mse.append(np.mean((linear_regression_model.predict(X_fold_val)*longevity_norm - y_fold_val*longevity_norm) ** 2))
                lasso_mse.append(np.mean((lasso_model.predict(X_fold_val)*longevity_norm - y_fold_val*longevity_norm) ** 2))
                ridge_mse.append(np.mean((ridge_model.predict(X_fold_val)*longevity_norm - y_fold_val*longevity_norm) ** 2))
                elastic_net_mse.append(np.mean((elastic_net_model.predict(X_fold_val)*longevity_norm - y_fold_val*longevity_norm) ** 2))
            else:
                linear_regression_mse.append(np.mean((linear_regression_model.predict(X_fold_val)*longevity_std+longevity_mean - y_fold_val*longevity_std+longevity_mean) ** 2))
                ridge_mse.append(np.mean((ridge_model.predict(X_fold_val)*longevity_std+longevity_mean - y_fold_val*longevity_std+longevity_mean) ** 2))
                lasso_mse.append(np.mean((lasso_model.predict(X_fold_val)*longevity_std+longevity_mean - y_fold_val*longevity_std+longevity_mean) ** 2))
                elastic_net_mse.append(np.mean((elastic_net_model.predict(X_fold_val)*longevity_std+longevity_mean - y_fold_val*longevity_std+longevity_mean) ** 2))

        linear_regression_real_scale_val = np.mean(linear_regression_mse)
        lasso_real_scale_val = np.mean(lasso_mse)
        ridge_real_scale_val = np.mean(ridge_mse)
        elastic_net_real_scale_val = np.mean(elastic_net_mse)

    if classification_step:
        log_name = 'log_hyperparameters_cls.csv'
    else:
        log_name = 'log_hyperparameters_reg.csv'

    hyperparameters_cls_names = ['keep_incomplete_data',
                                 'normalize_data',
                                 'standardize_data',
                                 'full_one_hot',
                                 'polynomial_degree',
                                 'logistic_penalty',
                                 'logistic_C',
                                 'logistic_solver',
                                 'logistic_multiclass',
                                 'random_forest_n_estimators',
                                 'random_forest_criterion',
                                 'knn_n_neighbours',
                                 'knn_algorithm',
                                 'knn_p']

    hyperparameters_reg_names = ['keep_incomplete_data',
                                 'normalize_data',
                                 'standardize_data',
                                 'full_one_hot',
                                 'polynomial_degree',
                                 'reg_fit_intercept',
                                 'lasso_alpha',
                                 'ridge_alpha',
                                 'ridge_solver',
                                 'el_net_alpha',
                                 'el_net_l1_ratio']

    results_cls_names = ['Logistic_Acuracy',
                         'KNN_Acuracy',
                         'Random_forest_Acuracy',
                         'Logistic_Precision',
                         'KNN_Precision',
                         'Random_forest_Precision',
                         'Logistic_F1',
                         'KNN_F1',
                         'Random_forest_F1']

    results_reg_names = ['linear_regression_mse',
                         'lasso_mse',
                         'ridge_mse',
                         'elastic_net_mse']

    if classification_step:
        names_list = hyperparameters_cls_names[:]
        names_list.extend(results_cls_names)
    else:
        names_list = hyperparameters_reg_names[:]
        names_list.extend(results_reg_names)

    try:
        log = pandas.read_csv(log_name, index_col='index')
    except FileNotFoundError:
        log = pandas.DataFrame(columns=names_list)

    if classification_step:
        new_log_cls_row = pandas.Series([keep_incomplete_data,
                                        normalize_data,
                                        standardize_data,
                                        full_one_hot,
                                        polynomial_degree,
                                        logistic_penalty,
                                        logistic_C,
                                        logistic_solver,
                                        logistic_multiclass,
                                        random_forest_n_estimators,
                                        random_forest_criterion,
                                        knn_n_neighbours,
                                        knn_algorithm,
                                        knn_p,
                                        np.mean(logistic_regression_accuracy),
                                        np.mean(knn_accuracy),
                                        np.mean(random_forest_accuracy),
                                        np.mean(logistic_regression_precision),
                                        np.mean(knn_precision),
                                        np.mean(random_forest_precision),
                                        np.mean(logistic_regression_f1),
                                        np.mean(knn_f1),
                                        np.mean(random_forest_f1)], index=log.columns)
    else:
        new_log_reg_row = pandas.Series([keep_incomplete_data,
                                        normalize_data,
                                        standardize_data,
                                        full_one_hot,
                                        polynomial_degree,
                                        reg_fit_intercept,
                                        lasso_alpha,
                                        ridge_alpha,
                                        ridge_solver,
                                        el_net_alpha,
                                        el_net_l1_ratio,
                                        linear_regression_real_scale_val,
                                        lasso_real_scale_val,
                                        ridge_real_scale_val,
                                        elastic_net_real_scale_val], index=log.columns)

    if classification_step:
        log = log.append(new_log_cls_row, ignore_index=True)
        log = log.drop_duplicates(subset=hyperparameters_cls_names)
    else:
        log = log.append(new_log_reg_row, ignore_index=True)
        log = log.drop_duplicates(subset=hyperparameters_reg_names)


    log.to_csv(log_name, columns=log.columns, index_label='index')


current_step = 0
classification_step = True
for combination_step in hyperparameters_combinations_cls:
    (keep_incomplete_data,
    normalize_data,
    standardize_data,
    full_one_hot,
    polynomial_degree,
    logistic_penalty,
    logistic_C,
    logistic_solver,
    logistic_multiclass,
    random_forest_n_estimators,
    random_forest_criterion,
    knn_n_neighbours,
    knn_algorithm,
    knn_p) = combination_step

    current_step = current_step + 1
    print(current_step, "/", len(hyperparameters_combinations_cls))

    if polynomial_degree > 5:
        continue
    if current_step % 20 == 0:
        gc.collect()

    run_grid()


current_step = 0
classification_step = False
for combination_step in hyperparameters_combinations_reg:
    (keep_incomplete_data,
    normalize_data,
    standardize_data,
    full_one_hot,
    polynomial_degree,
    reg_fit_intercept,
    lasso_alpha,
    ridge_alpha,
    ridge_solver,
    el_net_alpha,
    el_net_l1_ratio) = combination_step

    current_step = current_step + 1
    print(current_step, "/", len(hyperparameters_combinations_reg))

    if full_one_hot and polynomial_degree > 5:
        continue
    if current_step % 20 == 0:
        gc.collect()

    run_grid()
