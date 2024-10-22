import pandas as pd
import numpy as np
from DecisionTreeClassifier import Node, DecisionTreeClassifier


class RandomForestClassifier:
    def __init__(self, n_estimator = 100, max_depth = None, max_features = "sqrt", max_samples = None):
        self.n_estimator = n_estimator
        self.max_depth = max_depth
        self.max_features = "sqrt"
        self.max_samples = max_samples
        self.estimators = []
        
        for i in range(n_estimator):
            DTC = DecisionTreeClassifier(max_depth=max_depth, max_features=max_features)
            self.estimators.append(DTC)

    def sample(self, X, y):
        if self.max_samples is None:
            n_samples = X.shape[0] # row
        else:
            n_samples = min(self.max_samples, X.shape[0])
        
        indices = np.random.choice(X.shape[0], n_samples, replace = True)
        return X[indices], y[indices]

    def fit(self, X, y):
        for i in range(self.n_estimator):
            X_sample, y_sample = self.sample(X, y) # max_samples
            self.estimators[i].fit(X_sample, y_sample) # DTC.fit
    
    def predict(self, X):
        all_pred = np.zeros((self.n_estimator,
                            X.shape[0]),
                            dtype=np.int64)
        for i in range(self.n_estimator):
            all_pred[i] = self.estimators[i].predict(X) # DTC.predict
        preds = np.array([np.bincount(all_pred[:,i]).argmax() for i in range(X.shape[0])])
        # bincount of estimator's prediction array
        return preds


if __name__ == "__main__":
    train = pd.read_csv('./titanic/train.csv')
    
    train['Age'] = train['Age'].fillna(train['Age'].mean())
    feature = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex']
    X = pd.get_dummies(train[feature], drop_first = True).values
    y = train['Survived'].values
    
    RFC = RandomForestClassifier(n_estimator=50,
                                max_depth = 3,
                                max_features = 3)
    RFC.fit(X,y)
    y_pred = RFC.predict(X)
    
    print('train_acc :', (y_pred == y).mean())
    
    