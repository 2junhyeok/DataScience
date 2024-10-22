import pandas as pd
import numpy as np

class Node:
    def __init__(self, gini, n_samples, n_samples_per_class, predicted_class):
        self.gini = gini
        self.n_samples = n_samples
        self.n_samples_per_class = n_samples_per_class
        self.predicted_class = predicted_class
        self.feature_index = -1
        self.threshold = -1
        self.left = None
        self.right = None
        

class DecisionTreeClassifier():
    def __init__(self, max_depth=3, max_features = None):
        self.max_depth = max_depth
        self.tree = None
        self.max_features = max_features
        
    def fit(self, X, y):
        self.tree = self.build_tree(X, y, self.max_depth)
        
    def predict(self, X):
        return np.array([self.predict_one(x) for x in X])
    
    def gini(self, y):
        n_samples = len(y)
        n_samples_per_class = np.bincount(y)
        gini = 1.0
        for i in range(len(n_samples_per_class)):
            gini -= (n_samples_per_class[i] / n_samples) ** 2
        return gini
    
    def feature_samples(self, X, y):
        if self.max_features in None:
            max_features = X.shape[1]
        else:
            max_features = min(self.max_features, X.shape[1])
            
        indices = np.random.choice(X.shape[1], 
                            max_features, 
                            replace = True)
        return X.columns[indices]

    def find_best_split(self, X, y):
        # 가장 gini impurity를 낮게 해주는
        n_samples, n_features = X.shape
        best_gini = 1.0
        best_feature = -1
        best_threshold = -1
        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_indices = X[:, feature] < threshold # 기준보다 작은애들
                right_indices = ~left_indices
                left_gini = self.gini(y[left_indices])
                right_gini = self.gini(y[right_indices])
                gini = (left_gini * np.sum(left_indices) + right_gini * np.sum(right_indices)) / n_samples
                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature
                    best_threshold = threshold
        return best_feature, best_threshold
    
    def build_tree(self, X, y, depth):
        
        if depth < 0:
            return None
        
        n_samples_per_class=np.bincount(y)
        node = Node(
                    gini=self.gini(y),
                    n_samples=len(y),
                    n_samples_per_class=n_samples_per_class,
                    predicted_class=np.argmax(n_samples_per_class)
                    )

        if node.gini == 0:
            return node
        
        best_feature, best_threshold = self.find_best_split(X, y)

        node.feature_index = best_feature
        node.threshold = best_threshold
        
        left_indices = X[:, best_feature] < best_threshold
        right_indices = ~left_indices
        
        node.left = self.build_tree(X[left_indices], y[left_indices], depth - 1)
        node.right = self.build_tree(X[right_indices], y[right_indices], depth - 1)
        
        return node
    def predict(self, X):
        return np.array([self.predict_one(x) for x in X])

    def predict_one(self, x):
        node = self.tree
        while node.left is not None:
            if x[node.feature_index] < node.threshold:
                node = node.left
            else:
                node = node.right
        return node.predicted_class


if __name__ == "__main__":
    train = pd.read_csv('./titanic/train.csv')
    train.Age = train.Age.fillna(train.Age.mean())
    
    feature = ['Sex', 'Pclass', 'SibSp', 'Parch']
    
    X = pd.get_dummies(train[feature], drop_first = True)
    X.Sex_male = X.Sex_male.astype('int')
    
    y = train.Survived # np.array
    
    model = DecisionTreeClassifier()
    model.fit(X.values, y.values)
    print("train_acc :",(model.predict(X.values)==y.values).mean())