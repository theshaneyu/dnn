import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelBinarizer
from pprint import pprint


class iris_dnn(object):
    def __init__(self, data_path='./data.csv'):
        self.data = pd.read_csv(data_path, header=None)
        self.seed = 7
        # fix random seed for reproducibility
        np.random.seed(self.seed)

    def _preprocessing(self):
        dataset = self.data.values
        data = dataset[:, 0:4]
        label = dataset[:, 4]
        label = self._label_data_onehot_encoding(label)
        return data, label

    def _label_data_onehot_encoding(self, label):
        encoder = LabelBinarizer()
        encoder.fit(label)
        label = encoder.transform(label)
        return label

    # define baseline model
    def _baseline_model(self):
        # create model
        model = Sequential()
        model.add(Dense(8, input_dim=4, activation='relu'))
        model.add(Dense(3, activation='softmax'))
        # Compile model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def estimate_and_evaluate(self):
        # 以baseline_model函式作為model
        estimator = KerasClassifier(build_fn=self._baseline_model, epochs=200, batch_size=5, verbose=0)
        kfold = KFold(n_splits=10, shuffle=True, random_state=self.seed)
        
        data, label = self._preprocessing()

        results = cross_val_score(estimator, data, label, cv=kfold)
        print(results)
        print('Baseline Model 的平均正確率: %.2f%% ' % (results.mean()*100))
        print('Baseline Model 的正確率標準差: %.2f%% ' % (results.std()*100))


if __name__ == '__main__':
    iris_obj = iris_dnn()
    iris_obj.estimate_and_evaluate()
