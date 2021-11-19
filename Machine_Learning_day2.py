#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn import datasets
import numpy as np

iris = datasets.load_iris()
X = iris.data[:, [2,3]]
y = iris.target
print('클래스 레이블:',np.unique(y))


# In[5]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

# stratify = 계층화, 테스트데이터셋과 훈련데이터셋의 클래스 레이블 비율을 동일하게 만듬.
# 각 레이블의 빈도가 동일한지 확인하였음.
print('y의 레이블 카운트:', np.bincount(y))
print('y_train의 레이블 카운트:', np.bincount(y_train))
print('y_test의 레이블 카운트:', np.bincount(y_test))


# In[6]:


# 경사하강법을 위한 특성 스케일 조정

from sklearn.preprocessing import StandardScaler
sc = StandardScaler() # 새로운 객체를 sc에 할당하고,
sc.fit(X_train) # fit 메서드를 통해 X_train의 평균과 표준편차를 계산함.
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
# x_train과 X_test가 동일한 평균과 표준편차로 표준화한다,


# In[9]:


# 퍼셉트론 주입

from sklearn.linear_model import Perceptron
ppn = Perceptron(eta0 = 0.1, random_state=1) # max_iter = 훈련 에포크 횟수
ppn.fit(X_train_std, y_train)

y_pred = ppn.predict(X_test_std)
print('잘못 분류된 샘플 개수: %d' % (y_test != y_pred).sum())

# 오차는 약 2.2% 이다. 정확도는 97.8% 이다.


# In[10]:


# 사이킷런에서의 성능지표 구현
from sklearn.metrics import accuracy_score
print('정확도: %.3f' % accuracy_score(y_test, y_pred))


# In[22]:


a_ = np.arange(0,1,0.02)
b_ = np.arange(1,2,0.02)

aa_, bb_ = np.meshgrid(a_,b_)
np.array([aa_.ravel(), bb_.ravel()]).T
classifier.predict(np.array([aa_.ravel(), bb_.ravel()]).T)


# In[31]:


# 과대적합 = overfitting
# 데이터셋의 정확도가 너무 높으면, 새로운 데이터에 대한 정확도가 떨어진다.
# 반대로, underfitting 이라는 용어도 있다.
# 분류 시각화하기

from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    # 마커와 컬러맵을 설정합니다.
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    # 결정경계를 그립니다.
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
        plt.scatter(x=X[y == cl, 0], y=X[ y== cl, 1],
                    alpha=0.8, c=colors[idx],
                    marker=markers[idx], label=cl,
                    edgecolor='black')
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0], X_test[:, 1],
                   facecolors='none', edgecolor='black', alpha=1.0,
                   linewidth=1, marker='o',
                   s=100, label='test set')


# In[36]:


X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train,y_test))
plot_decision_regions(X=X_combined_std,
                      y=y_combined,
                     classifier=ppn,
                     test_idx=range(105, 150))
plt.xlabel('petal length [std]')
plt.ylabel('petal width [std]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
print("애초에 선형결정 경계가 명확하게 분류가 되지않는다.")
print("따라서 퍼셉트론 알고리즘은 수렴하지못하며, 실전에서 잘 추천되지않는다.")


# In[44]:


# 선형 이진 문제에 더 강혁한 로지스틱 회귀 알고리즘
# 로지스틱 회귀는 회귀가 아니라 분류 모델이다.
# 특성의 가중치 합과 로그 오즈 사이의 선형 관계 설명에 앞서,
# 로짓 함수(로그 오즈)를 뒤집은 시그모이드 함수를 확인해보기위해, 그래프를 그려본다.

import matplotlib.pyplot as plt
import numpy as np
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

z = np.arange(-7,7,0.1)
phi_z = sigmoid(z)
plt.plot(z, phi_z)
plt.axvline(0.0, color='k')
plt.ylim(-0.1, 1.1)
plt.xlabel('z')
plt.ylabel('$\phi (z)$')
# y축의 눈금과 격자선
plt.yticks([0.0, 0.5, 1.0])
ax = plt.gca()
ax.yaxis.grid(True)
plt.tight_layout()
plt.show()


# In[47]:


# 임계함수의 결과에 따른 시그모이드 활성화 함수의 비용 그래프
# 임계함수 값 y = 1 은 일치, y = 0 은 불일치를 의미한다.

def cost_1(z):
    return - np.log(sigmoid(z))

def cost_0(z):
    return - np.log(1-sigmoid(z))

z = np.arange(-10, 10, 0.1)
phi_z = sigmoid(z)

c1 = [cost_1(x) for x in z]
plt.plot(phi_z, c1, label='j(w) y=1 일때')
c0 = [cost_0(x) for x in z]
plt.plot(phi_z, c0, label='j(w) y=0 일때')
plt.ylim(0.0, 5.1)
plt.xlim([0, 1])
plt.legend(loc='best')
plt.tight_layout()
plt.show()

print("정확하게 예측한 경우, 비용함수의 값(y축)이 급격하게 올라가는 것을 알 수 있다.")


# In[55]:


# 아달린 구현을 로지스틱 회귀 알고리즘으로 변경
# 로지스틱 회귀 모델을 구현하려면, 아달린 구현모델에서 비용 함수를 변경 해주기만하면 된다.

class LogisticRegressionGD(object):
    def __init__(self, eta=0.05, n_iter=100, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        
    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size = 1 + X.shape[1])
        self.cost_ = []
        
        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y-output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            
            cost = (-y.dot(np.log(output)) -
                   ((1 - y).dot(np.log(1 - output))))
            self.cost_.append(cost)
        return self
    
    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def activation(self, z):
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))
    
    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, 0)
    # return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)


# In[58]:


# 로지스틱 회기 분석 결정 경계 시각화
X_train_01_subset = X_train[(y_train == 0) | (y_train == 1)]
y_train_01_subset = y_train[(y_train == 0) | (y_train == 1)]

lrgd = LogisticRegressionGD(eta = 0.05,
                            n_iter = 1000,
                            random_state=1)

lrgd.fit(X_train_01_subset,
         y_train_01_subset)

plot_decision_regions(X=X_train_01_subset,
                      y=y_train_01_subset,
                      classifier = lrgd)
plt.legend(loc='best')
plt.show()


# In[60]:


# 사이킷 런을 활용하여 로지스틱 회귀 모델 훈련

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(C=100.0, random_state=1)
# 매배견수 C는 과대적합을 막기 위한 규제와 관련된 매개변수이다.

lr.fit(X_train_std, y_train)
plot_decision_regions(X_combined_std,
                      y_combined,
                     classifier = lr,
                     test_idx=range(105,150))
plt.xlabel('petal length [std]')
plt.ylabel('petal width [std]')
plt.legend(loc='best')
plt.tight_layout()
plt.show()


# In[67]:


lr.predict_proba(X_test_std[:,:])
# 첫 번째 행은 첫번째 붗꽃의 클래스 소속 활률임.
# 열을 모두 더하면 1이 된다.
# 여기 확률중에서 가장 높은 확률이 예측 클래스의 레이블이 된다.


# In[72]:


# Underfitting 과 overfitting
# 규제를 사용하여 overfitting을 피하기
# 편향 분산 트레이드 오프 : 
# 편향을 줄이면 분산이 높아지고(overfitting), 분산을 낮추려면 편향이 높아진다.(underfitting)
# 좋은 편향과 분산 트레이드오프를 찾는 방법 중하나는 규제를 사용하여 모델의 복잡도를 조정하는 것임.
# 규제는 공선성(특성간 너무 높은 상관관계)를 다루거나, 데이터에서 잡음을 제거하여 과대적합을
# 방지하는 매우 유영한 방법임.
# 규제를 사용하기 위해서는, 특성의 표준화 전처리가 반드시 필요함.

weights, params = [], []
for c in np.arange(-5,5):
    lr = LogisticRegression(C=10.**c, random_state=1, multi_class='ovr')
    lr.fit(X_train_std, y_train)
    weights.append(lr.coef_[1])
    params.append(10.**c)

weights = np.array(weights)
plt.plot(params, weights[:, 0],
         label='petal length')
plt.plot(params, weights[:, 1],
         label='petal width')
plt.legend(loc='best')
plt.xscale('log')
plt.xlabel('C')
plt.ylabel('weight coefficient')
plt.show()
print('매개변수 C가 감소하면 규제강도가 높아져 가중치의 변화가 거의 없어진다.')
print('반대로 매개변수 C가 늘어나면 규제 강도가 낮아서 가중치의 변화가 커진다.')


# In[81]:


# 서포트 벡터머신을 사용한 최대 마진 분류
# 서포트 벡터 머신(Support vector machine, SVM)
# 마진 : 클래스를 구분하는 초평면과 이 초평면에 가장 가까운 훈련샘플 사이의 거리
# 최대 마진 : 일반화의 오차가 낮아지는 경향이 있기 때문에, 높은 마진의 결정 경계를 원한다.
# 매개변수C와 상관관계 : 매개변수 C가 클때(규제약화) 마진의 경계는 엄격해진다.
# 뭔소린지 모르겠다.

from sklearn.svm import SVC
svm = SVC(kernel = 'linear', C=1.0, random_state=1)
svm.fit(X_train_std, y_train)
plot_decision_regions(X_combined_std,
                      y_combined,
                      classifier=svm,
                      test_idx=range(105,150))
plt.xlabel('petal length(std)')
plt.ylabel('petal width(std)')
plt.legend(loc='best')
plt.tight_layout()
plt.show()
# 로지스틱 회귀는 훈련데이터의 조건부 가능도를 최대화하기때문에 SVM보다 이상치에 민감하다.
# SVM은 결정 경계에 가장 가까운 포인트(서포트벡터)에 관심을 둔다.
print('로지스틱 회귀분석과 비슷한 결과가 나왔다. ')


# In[84]:


np.random.randn(200,2)


# In[85]:


# 비선형적 데이터를 위한 커널방법
# 커널 SVM을 사용하여 비선형 문제 풀기

import matplotlib.pyplot as plt
import numpy as np
np.random.seed(1)
X_xor = np.random.randn(200,2)
y_xor = np.logical_xor(X_xor[:, 0] > 0,
                       X_xor[:, 1] > 0)
y_xor = np.where(y_xor, 1, -1)
plt.scatter(X_xor[y_xor == 1, 0],
            X_xor[y_xor == 1, 1],
            c='b', marker='x',
            label='1')
plt.scatter(X_xor[y_xor == -1, 0],
            X_xor[y_xor == -1, 1],
            c='r', marker='s',
            label='-1')
plt.xlim([-3,3])
plt.ylim([-3,3])
plt.legend(loc='best')
plt.tight_layout()
plt.show()

# 2차원 평면에서 선형적으로 구분이 불가능한 데이터셋 그래프가 나옴.
# 이를 해결하기위해 커널 기법을 사용하여 고차원 공간에서 분할 초평면 찾기


# In[88]:


# 커널 함수 : 방사기저 함수 (가우시안 커널)
# 커널 = 샘플간의 유사도 함수

svm = SVC(kernel='rbf', random_state=1, gamma=0.10, C=10.0)
svm.fit(X_xor, y_xor)
plot_decision_regions(X_xor, y_xor, classifier=svm)
plt.legend(loc='best')
plt.show()


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
