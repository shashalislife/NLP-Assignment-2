###Part 1
"""
1. Re-code the house price machine learning
1. Random Choose Method to get optimal k and b
2.Supervised Direction to get optimal k and b
3.Gradient Descent to get optimal k and b
4. Try different Loss function and learning rate.
For example, you can change the loss function: $Loss = \frac{1}{n} sum({y_i - \hat{y_i}})^2$ to $Loss = \frac{1}{n} sum(|{y_i - \hat{y_i}}|)$
"""

from sklearn.datasets import load_boston
import matplotlib.pyplot as plt
import random

data=load_boston()

X,y=data['data'],data['target']

X_rm=X[:,5]
#print(data.keys())
# print(data['feature_names'])
# print(data['DESCR'])

#print(len(X),len(y))

# def draw_rm_and_price():
#     plt.scatter(X[:,5],y)
#     plt.show()

def price(rm,k,b):
    return k*rm +b

def loss(y, y_hat):
    return sum((y_i - y_hat_i)**2 for y_i, y_hat_i in zip(list(y),list(y_hat)))/ len(list(y))

### 1. random choose Methode to geht optimal k and b, get the best value of the loss function

# trying_times=1000
#
# min_loss=float('inf')
# ### 问题1：inf 的作用是什么？
# best_k, best_b= None, None
#
# for i in range(trying_times):
#     #Initialization
#     k= random.random()*200-100
#     b=random.random()*200-100
#     price_by_random_k_and_b = [price(r, k, b)for r in X_rm]
#
#     current_loss=loss(y, price_by_random_k_and_b)
#
#     if current_loss < min_loss: # performance became better
#         min_loss = current_loss
#         best_k, best_b = k, b
#         print('When time ist : {}, get best_k: {}, best_b: {}, and the loss is: {}'.format(i, best_k, best_b, min_loss))


### 2. Supervised Direction to get optimal k and b
# trying_times = 2000
# min_loss = float('inf')
# best_k = random.random() * 200 - 100
# best_b = random.random() * 200 - 100
#
# direction = [
#     (+1, -1),  # first element: k's change direction, second element: b's change direction
#     (+1, +1),
#     (-1, -1),
#     (-1, +1),
# ]
#
# next_direction = random.choice(direction)
# scalar = 0.1
# update_time = 0
#
# for i in range(trying_times):
#     k_direction, b_direction = next_direction
#     current_k, current_b = best_k + k_direction * scalar, best_b + b_direction * scalar
#     price_by_k_and_b = [price(r, current_k, current_b) for r in X_rm]
#     current_loss = loss(y, price_by_k_and_b)
#
#     if current_loss < min_loss:  # performance became better
#         min_loss = current_loss
#         best_k, best_b = current_k, current_b
#
#         next_direction = next_direction
#         update_time += 1
#
#         if update_time % 20 == 0:
#             print(
#                 'When time is : {}, get best_k: {} best_b: {}, and the loss is: {}'.format(i, best_k, best_b, min_loss))
#     else:
#         next_direction = random.choice(direction)

####3.Gradient Descent to get optimal k and b

def partial_k(x,y,y_hat):
    n=len(y)
    gradient=0
    for x_i, y_i, y_hat_i in zip(list(x),list(y),list(y_hat)):
        gradient += (y_i - y_hat_i)*x_i
    return -2 / n*gradient


def partial_b(x, y, y_hat):
    n = len(y)
    gradient = 0
    for y_i, y_hat_i in zip(list(y), list(y_hat)):
        gradient += (y_i - y_hat_i)
    return -2 / n * gradient

trying_times = 2000
min_loss = float('inf')
current_k = random.random() * 200 - 100
current_b = random.random() * 200 - 100
learning_rate = 1e-04
update_time = 0

for i in range(trying_times):
    price_by_k_and_b=[price(r, current_k, current_b) for r in X_rm]
    current_loss=loss(y, price_by_k_and_b)

    if current_loss < min_loss:
        min_loss = current_loss


        k_gradient=partial_k(X_rm, y, price_by_k_and_b)
        b_gradient=partial_b(X_rm, y, price_by_k_and_b)

        current_k=current_k+(-1*k_gradient)*learning_rate
        current_b=current_b+(-1*b_gradient)*learning_rate

        if i % 50 == 0:
            print('When time is : {}, get best_k: {} best_b: {}, and the loss is: {}'.format(i, current_k, current_b, min_loss))
        ##问题2：老师的代码，打印一项为什么在更新值之前？虽然貌似不影响结果。