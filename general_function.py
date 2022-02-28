from pandas import read_csv
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from imblearn.over_sampling import ADASYN


def load_data(file_name, num_of_var):
    df = read_csv(file_name, delimiter=';')

    x = df.iloc[:, 0:num_of_var]
    y = df.iloc[:, num_of_var]

    return x, y


def encode_data(x, y):
    encoded_data = None
    for col in x:
        label_encoder = LabelEncoder()
        feature = label_encoder.fit_transform(x[col])
        feature = feature.reshape(x.shape[0], 1)
        column = [col]
        feature_df = pd.DataFrame(feature, columns=column)
        # onehot_encoder = OneHotEncoder(sparse=False, categories='auto')
        # feature = onehot_encoder.fit_transform(feature)
        if encoded_data is None:
            encoded_data = feature_df
        else:

            encoded_data = pd.concat((encoded_data, feature_df), axis=1)

    label = label_encoder.fit_transform(y)

    data = encoded_data + label
    print(data.head)

    return encode_data, label


def feature_selection_using_tree(x, y):
    # define the model
    tree_model = DecisionTreeClassifier()
    # fit the model
    tree_model.fit(x, y)
    # get importance
    importance = tree_model.feature_importances_
    # summarize feature importance
    for i, v in enumerate(importance):
        print('Feature: %0d, Score: %.5f' % (i, v))
    # plot feature importance
    pyplot.bar([x for x in range(len(importance))], importance)
    pyplot.show()

    print('Before selection: ', x.shape)

    feature_select_model = SelectFromModel(tree_model, prefit=True)
    x = feature_select_model.transform(x)

    print('After selection: ', x.shape)
    return x


def feature_extraction_using_pca(x, y, num_of_var, threshold):
    pca = PCA(n_components=num_of_var)
    principal_components = pca.fit_transform(x)
    principalDf = pd.DataFrame(data=principal_components)

    percent_variance = np.round(pca.explained_variance_ratio_ * 100, decimals=2)
    plt.bar(x=range(0, 18), height=percent_variance, tick_label=principalDf.columns)
    plt.ylabel('Percentate of Variance Explained')
    plt.xlabel('Principal Component')
    plt.title('PCA Scree Plot')
    print(percent_variance)
    plt.show()

    total_variance = 0
    extracted_feature = None
    for i in percent_variance:
        if total_variance > threshold:
            break

        total_variance = total_variance + i
        extracted_feature = extracted_feature + principalDf[i]

    return extracted_feature


def oversampling_adasyn(x, y):
    print('Before Resample: ', y.value_counts())
    oversample = ADASYN(sampling_strategy='minority', random_state=8, n_neighbors=3)
    x, y = oversample.fit_resample(x, y)
    print('After Resample: ', y.value_counts())


def generate_report(y_train, y_test, y_pred_train, y_pred_test):
    print(classification_report(y_pred_train, y_train))

    cm = confusion_matrix(y_pred_train, y_train)
    acc = cm.diagonal().sum() / cm.sum()
    print("Training Acc: ", acc)

    cm = confusion_matrix(y_pred_test, y_test)
    acc = cm.diagonal().sum() / cm.sum()
    print("Testing Acc: ", acc)

    auc = roc_auc_score(y_pred_train, y_train)
    fpr, tpr, _ = roc_curve(y_pred_train, y_train)
    plt.plot(fpr, tpr, label="AUC=" + str(auc))
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.title('Training')
    plt.legend(loc=4)
    plt.show()

    auc = roc_auc_score(y_pred_test, y_test)
    fpr, tpr, _ = roc_curve(y_pred_test, y_test)
    plt.plot(fpr, tpr, label="AUC=" + str(auc))
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.title('Testing')
    plt.legend(loc=4)
    plt.show()
