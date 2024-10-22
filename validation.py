import numpy as np

def cross_val_score(model, X, y, fold):
    n = len(X)
    perm = np.random.permutation(n)
    X = X[perm]
    y = y[perm]
    
    scores = []
    
    for i in range(fold):
        start = i*(n//fold)
        end = (i+1)*(n//fold)
        
        X_train = np.concatenate([X[:start], X[end:]]) # 1 fold
        y_train = np.concatenate([y[:start], y[end:]]) # 1 fold
        
        X_test = X[start:end] # k-1 fold
        y_test = y[start:end] # k-1 fold
        
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        scores.append((pred==y_test).mean())
    return np.array(scores)