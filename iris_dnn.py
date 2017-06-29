import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
# from sklearn.pipeline import Pipeline


class iris_dnn(object):
    def __init__(self, data_path='./data.csv'):
        self.data = pd.read_csv(data_path, header=None)
        self.seed = 7

    def preprocessing(self):
        # fix random seed for reproducibility
        # numpy.random.seed(seed)
        dataset = self.data.values
        data = dataset[:, 0:4]
        label = dataset[:, 4]
        label = self.label_data_onehot_encoding(label)
        return data, label

    def label_data_onehot_encoding(self, label):
        # 將label encode成整數
        encoder = LabelEncoder()
        encoder.fit(label)
        encoded_label = encoder.transform(label) # 回傳一個numpy array，裡面是0, 1, 2 ...
        # 做one hot encoding
        one_hot_encoded_label = np_utils.to_categorical(encoded_label)
        # 0會變成[1 0 0], 1會變成[0 1 0] ...
        return one_hot_encoded_label

    # define baseline model
    def baseline_model(self):
        # create model
        model = Sequential()
        model.add(Dense(8, input_dim=4, activation='relu'))
        model.add(Dense(3, activation='softmax'))
        # Compile model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def estimate_and_evaluate(self):
        # 以baseline_model函式作為model
        estimator = KerasClassifier(build_fn=self.baseline_model, epochs=200, batch_size=5, verbose=0)
        kfold = KFold(n_splits=10, shuffle=True, random_state=self.seed)

        data, label = self.preprocessing()
        
        results = cross_val_score(estimator, data, label, cv=kfold)
        print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))






if __name__ == '__main__':
    iris_obj = iris_dnn()
    iris_obj.estimate_and_evaluate()