#!/usr/bin/env python
# coding: utf-8

# In[279]:


# 퍼셉트론 구현

import numpy as np

class Perceptron(object):
    """
    매개변수
    ------------------
    eta : float
     학습률 (0.0 ~ 1.0 사이)
    n_iter : int
     훈련데이터셋 반복횟수
    random_state : int
     가중치 무작위 초기화를 위한 난수 생성기 시드
     속성
     ----------------
     w_ : 1d_array
      학습된 가중치
     errors_ : list
      에포크마다 누적된 분류 오류"""
    
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        
    def fit(self, X, y):
        """
        매개변수
        -----------------
        X = {array-like}, shape = [n_samples, n_features]
         n_samples개의 샘플과 n_features개의 특정으로 이루어진 훈련데이터
        y = array-like, shape = [n_samples]
         타깃 값

        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01,
                              size=1 + X.shape[1]) # 가중치 초기화
        self.errors_ = []
        
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X,y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self
    
    def net_input(self, X):
        """입력계산"""
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def predict(self, X):
        """단위 계단 함수를 사용하여 클래스 레이블을 반환합니다."""
        return np.where(self.net_input(X) >= 0.0, 1, -1)
    


# In[11]:


import os
import pandas as pd


# In[346]:


df = pd.read_csv('./iris.data', header=None, encoding='utf-8')
df.tail()

import matplotlib.pyplot as plt
import numpy as np

y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)

X  = df.iloc[0:100,[0,2]].values

plt.scatter(X[:50, 0], X[:50, 1],
           color = 'red', marker = 'o', label='setosa')
plt.scatter(X[50 : 100, 0], X[50:100, 1],
           color= 'blue', marker = 'x', label='versicolor')
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')
plt.show()


# In[22]:


ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(X,y)
plt.plot(range(1, len(ppn.errors_) + 1),
         ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of updates')
plt.show()

print("에포크가 6일때, 오류없이 완벽하게 분류함.")


# In[32]:


from matplotlib.colors import ListedColormap

def plot_decision_regions(X, y, classifier, resolution=0.02):
    
    # 마커와 컬러맵 설정
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    #결정경계를 그립니다.
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                   alpha = 0.8,
                   c=colors[idx],
                   marker=markers[idx],
                   label=cl,
                   edgecolor='black')


# In[33]:


plot_decision_regions(X, y, classifier=ppn)
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')
plt.show()


# In[38]:


rgen = np.random.RandomState(1)
rgen.normal(loc=0.0, scale=0.01,
                              size=1 + X.shape[1])


# In[347]:


# 파이썬으로 아달린모댈 구현

class AdalineGD(object):
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        
    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01,
                              size=1 + X.shape[1])
        self.cost_ = []
        
        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self
    
    def net_input(self, X):
        """최종 입력 계산"""
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def activation(self, X):
        """선형 활성화 계산
        -> 항등함수이기때문에 아무런 영향을 미치지못함."""
        return X
    
    def predict(self, X):
        """단위 계단 함수를 사용하여 클래스 레이블을 반환합니다."""
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)


# In[349]:


# 학습률에 따른 에포크 횟수 대비 비용 그래프 그려보기
# 비용 = 목적함수 / 비용을 낮춰야 정확도가 높은 것이다.

fig, ax = plt.subplots(nrows = 1, ncols=2, figsize=(10,4))

ada1 = AdalineGD(n_iter=10, eta=0.01).fit(X,y)
ax[0].plot(range(1, len(ada1.cost_) + 1), np.log10(ada1.cost_), marker='o')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(Sum-squared-error)')
ax[0].set_title('Adaline - Learning rate 0.01')

ada2 = AdalineGD(n_iter=10, eta=0.0001).fit(X,y)
ax[1].plot(range(1, len(ada2.cost_) + 1), ada2.cost_, marker='o')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Sum-squared-error')
ax[1].set_title('Adaline - Learning rate 0.0001')

plt.show()


# In[67]:


# 데이터를 표준화를 하게되면, 경사하강법 학습이 좀 더 빠르게 이루어진다.
# 그 이유중 하나는, 더 적은 단계를 거쳐 최적 혹은 좋은 솔루션을 찾기 때문이다.
# 표준화, 각 특성 값의 평균을 빼고, 표준편차로 나누면 된다.

X_std = np.copy(X)
X_std[:, 0] = ( X_std[:, 0] - X_std[:, 0].mean()) / X_std[:, 0].std()
X_std[:, 1] = ( X_std[:, 1] - X_std[:, 1].mean()) / X_std[:, 1].std()

print(X_std[:5])


# In[72]:


# 표준화한 데이터를 가지고, 다시 아달린 모델을 훈련하고 학습률 0.01에서
# 몇번의 에포크만에 수렴하는지 확인

ada = AdalineGD(n_iter=15, eta=0.01)
ada.fit(X_std, y)

plot_decision_regions(X_std, y, classifier=ada)
plt.title('Adaline - Gradient Descent')
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

plt.plot(range(1, len(ada.cost_) +1), ada.cost_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Sum-squared-error')
plt.tight_layout()
plt.show()

print("동일한 학습률 0.01에서 표준화를 거치니 아달린 모델이 수렴하여 비용이 감소함.")


# In[167]:


w_ = rgen.normal(loc=0, scale=0.01, size=1 + X.shape[1])

np.dot(X, w_[1:]) + w_[0]


# In[170]:


# 확률적 경사 하강법 / 미니배치 경사하강법
# 확률적 경사 하강법 = Stochastic gradient descent

class AdalineSGD(object):
    """
    eta = 학습률
    n_iter = 훈련 횟수
    shuffle = True 일경우, 같은 반복이 일어나지않도록 에포크마다 훈련 데이터를 섞음
    random_state = 무작위 초기화
    
    w_ = 학습된 가중치
    cost_ = 모든 훈련 샘플에 대해 에포크마다 누적된 평균 비용 함수의 제곱합
    """
    
    def __init__(self, eta = 0.01, n_iter=10,
                 shuffle = True, random_state=None):
        self.eta = eta
        self.n_iter = n_iter
        self.shuffle = shuffle
        self.w_initialized = False
        self.random_state = random_state
        
    def fit(self, X, y):
        self._initialize_weights(X.shape[1]) # 가중치 초기화
        self.cost_ = []
        for i in range(self.n_iter): 
            if self.shuffle: # True일 경우, 훈련데이터를 섞음
                X,y = self._shuffle(X, y)
            cost = []
            
            for xi, target in zip(X,y):
                cost.append(self._update_weights(xi, target))
            avg_cost = sum(cost) / len(y)
            self.cost_.append(avg_cost)
        return self
    
    def partial_fit(self, X, y):
        """ 가중치를 다시 초기화하지않고 훈련 데이터를 학습함."""
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(X,y)
        return self
    
    def _shuffle(self, X, y):
        """훈련데이터 섞음"""
        r = self.rgen.permutation(len(y))
        return X[r], y[r]
    
    def _initialize_weights(self, m):
        """랜덤한 작은 수로 가중치를 초기화합니다."""
        self.rgen = np.random.RandomState(self.random_state)
        self.w_ = self.rgen.normal(loc=0, scale=0.01, size=1 + m)
        self.w_initialized = True # 원래는 False, -> True로 변경
    
    def _update_weights(self, xi, target):
        """아달린 학습 규칙을 적용하여 가중치를 업데이트함"""
        output = self.activation(self.net_input(xi))
        error = (target - output)
        self.w_[1:] += self.eta * xi.dot(error)
        self.w_[0] += self.eta * error
        cost = 0.5 * error **2
        return cost
    
    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def activation(self, X):
        return X
    
    def predict(self, X):
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)


# In[173]:


ada = AdalineSGD(n_iter=15, eta=0.01, random_state=1)
ada.fit(X_std,y)

plot_decision_regions(X_std, y, classifier=ada)
plt.title('Adaline - Stochastic Gradient Descent')
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.legend(loc='upper left')
plt.show()

plt.plot(range(1, len(ada.cost_) +1), ada.cost_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Average Cost')
plt.tight_layout()
plt.show()


# In[318]:


import seaborn as sns

df = sns.load_dataset('titanic')
df.head()
df.dropna(subset = ['age'], axis = 0, inplace= True)
df.drop(columns = ['deck'],inplace = True)
df[(df.fare < 200) & (df.pclass == 1)] = None
df[ df.pclass == 3] = None
df.dropna(inplace= True)
df = df[['pclass', 'age', 'fare']]
df = df[(df.fare > 200) | (df.fare < 13)]
df.pclass[ df.pclass == 1] = 1
df.pclass[ df.pclass == 2] = -1

df_first = df[df.pclass == 1]
df_second = df[df.pclass == -1]
df_third = df[df.pclass == 3]

df.info()


# In[319]:


df.head()


# In[320]:


plt.scatter(df_first.iloc[:,1], df_first.iloc[:,2],
           color = 'red', marker = 'o', label='First')
plt.scatter(df_second.iloc[:,1], df_second.iloc[:,2],
           color= 'blue', marker = 'x', label='Second')
# plt.scatter(df_third.iloc[:,1], df_third.iloc[:,2],
#            color= 'lightgreen', marker = 'x', label='Third')
plt.xlabel('Age')
plt.ylabel('Fare')
plt.legend(loc='upper left')
plt.show()


# In[322]:


df[['age','fare']].values


# In[332]:


X = df[['age','fare']].values
y = df.pclass.values

ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(X,y)
plt.plot(range(1, len(ppn.errors_) + 1),
         ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of updates')
plt.show()

print("에포크가 2일때, 오류없이 완벽하게 분류함.")


# In[339]:


plot_decision_regions(X, y, classifier=ppn)
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')
plt.show()
print("표준화를 거치지않고 학습한 결과")


# In[327]:


X_std = np.copy(X)
X_std[:, 0] = ( X_std[:, 0] - X_std[:, 0].mean()) / X_std[:, 0].std()
X_std[:, 1] = ( X_std[:, 1] - X_std[:, 1].mean()) / X_std[:, 1].std()


# In[331]:


ada = AdalineSGD(n_iter=15, eta=0.01, random_state=1)
ada.fit(X_std,y)

plot_decision_regions(X_std, y, classifier=ada)
plt.title('Adaline - Stochastic Gradient Descent')
plt.xlabel('Age [standardized]')
plt.ylabel('Fare [standardized]')
plt.legend(loc='upper left')
plt.show()

plt.plot(range(1, len(ada.cost_) +1), ada.cost_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Average Cost')
plt.tight_layout()
plt.show()

print("표준화를 거친 후, 아달린 모델의 확률적 경사 하강법으로 학습함.")


# In[345]:


ppn = Perceptron(eta=0.1, n_iter=10, random_state=None)
ppn.fit(X_std,y)
plt.plot(range(1, len(ppn.errors_) + 1),
         ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of updates')
plt.show()

print("에포크가 2일때, 오류없이 완벽하게 분류함.")

plot_decision_regions(X_std, y, classifier=ppn)
plt.title('Adaline - Stochastic Gradient Descent')
plt.xlabel('Age [standardized]')
plt.ylabel('Fare [standardized]')
plt.legend(loc='upper left')
plt.show()

print("다중 퍼셉트론 모델을 사용할때, 표준화를 거칠 경우 학습 속도가 획기적으로 빨라짐.")


# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
