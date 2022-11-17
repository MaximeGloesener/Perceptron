### Goal

The goal of this project is to provide a classifier for documents using a perceptron from scratch.
We have to compare GD/AGD/SGD and discuss effectiveness of these methods.

### Problem description

For each document x{i} (i=1,2,…,n), we are given its class f(i) in {1,2,…,r} where the function f : {1,…,n} --> {1,…,r} : i --> f(i) gives the topic of document i.

We want to find the parameters (W,b) that minimize

```
    sum_i L( phi(W^T x{i} + b) , y{i} ) [+ lambda ||W||F^2]
```

where

- y{i} = e{f(i)}, where e{j} is the jth unit vector (1 in position j, 0 otherwise),
- L(a,b) is the loss; for example, L(a,b) = ||a-b||^2 (least squares),
- phi(.) is the non-linear activation of the neural network; for example, phi(x) = max(0, x) (rectified linear unit ReLU), or phi(x) = 1 / (1 + exp(-a\*x) ), where a is a constant (logistic function).
