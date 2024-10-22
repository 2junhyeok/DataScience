import torch

# LinearRegression
def LinearRegression(X,y,
                    lr = 0.01,
                    epoch = 3000):
    
    w = torch.randn(X[0].size(0),1)
    b = torch.randn(1,1)

    
    for e in range(epoch): 
        w.requires_grad_(True)
        b.requires_grad_(True)
        
        h = torch.mm(X,w) + b
        cost = torch.mean((h-y)**2)
        
        # optimizer.zero_grad()
        if w.grad is not None:
            w.grad.zero_()
        if b.grad is not None:
            b.grad.zero_()
        
        cost.backward() # param.grad에 gradient 저장

        # 파라미터 수동 업데이트
        with torch.no_grad():
            # optimizer.step()
            w -= lr * w.grad  # w 업데이트
            b -= lr * b.grad  # b 업데이트
            
            if e % 500 == 0:
                print(f"Epoch: {e}, Cost: {cost.item()}, w: {w.squeeze()}, b: {b.item()}")
    
    return w, b

if __name__ == "__main__":
    X_train = torch.FloatTensor([[1,2], [3,2], [3,7], [1,1], [1,0]])
    y_train = torch.FloatTensor([[4], [8], [23], [1], [-2]])
    LinearRegression(X_train, y_train)
    
    