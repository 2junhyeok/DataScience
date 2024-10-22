import numpy as np
from validation import cross_val_score

class KNNClassifier:
    def __init__(self, k):
        self.k = k

    def fit(self, X, Y):
        self.X = X
        self.Y = Y

    def predict(self, queries):
        Y = []
        for q in queries:
            dists = np.linalg.norm(self.X - q, axis=1) 
            # 모든 X에 대해 query를 다 빼준다.
            # norm을 통해 모든 값과의 거리를 구해준다.
            knns = np.argsort(dists)[:self.k]
            # 거리기준 인덱스를 sort하고 k만큼 뽑는다.
            counts = np.bincount(self.Y[knns])
            # np.bincount(꽃 종류) -> 꽃 종류별 등장 횟수 count list
            Y.append(np.argmax(counts))
            # count list에서 최대값에 해당하는 레이블을 Y에 추가
            
        return np.array(Y)
    

if __name__ == "__main__":
    X = []
    y = []
    for line in open("iris.data", "r"):
        line = line.strip()
        if line != "":
            tokens = line.split(",")
            X.append([float(t) for t in tokens[:4]]) # 5.1, 2.2, 3.4, 5.2
            y.append(tokens[4]) # Iris-setosa
    labels = list(set(y)) # ['Iris-virginica', 'Iris-versicolor', 'Iris-setosa']
    
    y = [labels.index(i) for i in y] # label to idx
    
    X = np.array(X)
    y = np.array(y)
    
    model = KNNClassifier(4)
    scores = cross_val_score(model, X, y, 8)
    print(scores.mean())
    