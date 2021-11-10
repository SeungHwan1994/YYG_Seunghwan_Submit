#!/usr/bin/env python
# coding: utf-8

# In[21]:


# %load graphicsource_include.py
#!/usr/bin/env python

# In[1]:


# Seabron 세팅
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
# 한글 폰트 적용
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
# 마이너스 부호 적용
plt.rc('axes', unicode_minus=False)


# In[22]:


titanic.info()


# In[23]:


titanic = sns.load_dataset('titanic')
# seaborn regplot
# 스타일 테마 설정 (5가지: darkgrid, whitegrid, dark, white, ticks)
sns.set_style('darkgrid')

# 그래프 객체 생성 (figure에 2개의 서브 플롯을 생성)
fig = plt.figure(figsize=(15, 5))   
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)
 
# 그래프 그리기 - 선형회귀선 표시(fit_reg=True)
sns.regplot(x='age',        #x축 변수
            y='fare',       #y축 변수
            data=titanic,   #데이터
            ax=ax1)         #axe 객체 - 1번째 그래프 

# 그래프 그리기 - 선형회귀선 미표시(fit_reg=False)
sns.regplot(x='age',        #x축 변수
            y='fare',       #y축 변수
            data=titanic,   #데이터
            ax=ax2,         #axe 객체 - 2번째 그래프        
            fit_reg=False)  #회귀선 미표시

plt.show()


# In[24]:


# seaborn displot
# 그래프 객체 생성 (figure에 3개의 서브 플롯을 생성)
fig = plt.figure(figsize=(15, 5))   
ax1 = fig.add_subplot(1, 3, 1)
ax2 = fig.add_subplot(1, 3, 2)
ax3 = fig.add_subplot(1, 3, 3)
 
# distplot
sns.histplot(titanic['fare'], ax=ax1) 

# kdeplot
sns.kdeplot(x='fare', data=titanic, ax=ax2) 

# histplot
sns.histplot(x='fare', data=titanic,  ax=ax3)        

# 차트 제목 표시
ax1.set_title('titanic fare - distplot')
ax2.set_title('titanic fare - kedplot')
ax3.set_title('titanic fare - histplot')

plt.show()


# In[25]:


# seaborn heatmap
# 피벗테이블로 범주형 변수를 각각 행, 열로 재구분하여 정리
table = titanic.pivot_table(index=['sex'], columns=['class'], aggfunc='size')

# 히트맵 그리기
sns.heatmap(table,                  # 데이터프레임
            annot=True, fmt='d',    # 데이터 값 표시 여부, 정수형 포맷
            cmap='YlGnBu',          # 컬러 맵
            linewidth=.5,           # 구분 선
            cbar=False)             # 컬러 바 표시 여부

plt.show()


# In[26]:


# seaborn scatter
#스타일 테마 설정 (5가지: darkgrid, whitegrid, dark, white, ticks)
sns.set_style('whitegrid')

# 그래프 객체 생성 (figure에 2개의 서브 플롯을 생성)
fig = plt.figure(figsize=(15, 5))   
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)
 
# 이산형 변수의 분포 - 데이터 분산 미고려
sns.stripplot(x="class",      #x축 변수
              y="age",        #y축 변수           
              data=titanic,   #데이터셋 - 데이터프레임
              ax=ax1)         #axe 객체 - 1번째 그래프 

# 이산형 변수의 분포 - 데이터 분산 고려 (중복 X) 
sns.swarmplot(x="class",      #x축 변수
              y="age",        #y축 변수
              data=titanic,   #데이터셋 - 데이터프레임
              size=4,
              ax=ax2)         #axe 객체 - 2번째 그래프        

# 차트 제목 표시
ax1.set_title('Strip Plot')
ax2.set_title('Strip Plot')

plt.show()


# In[27]:


# seaborn bar
# 그래프 객체 생성 (figure에 3개의 서브 플롯을 생성)
fig = plt.figure(figsize=(15, 5))   
ax1 = fig.add_subplot(1, 3, 1)
ax2 = fig.add_subplot(1, 3, 2)
ax3 = fig.add_subplot(1, 3, 3)
 
# x축, y축에 변수 할당
sns.barplot(x='sex', y='survived', data=titanic, ax=ax1) 

# x축, y축에 변수 할당하고 hue 옵션 추가 
sns.barplot(x='sex', y='survived', hue='class', data=titanic, ax=ax2) 

# x축, y축에 변수 할당하고 hue 옵션을 추가하여 누적 출력
sns.barplot(x='sex', y='survived', hue='class', dodge=False, data=titanic, ax=ax3)       

# 차트 제목 표시
ax1.set_title('titanic survived - sex')
ax2.set_title('titanic survived - sex/deck') # sex 안에서 class 별로 구분
ax3.set_title('titanic survived - sex/class(stacked)') # sex안에서 class 누적 구분

plt.show()


# In[28]:


# seaborn count
# 그래프 객체 생성 (figure에 3개의 서브 플롯을 생성)
fig = plt.figure(figsize=(15, 5))   
ax1 = fig.add_subplot(1, 3, 1)
ax2 = fig.add_subplot(1, 3, 2)
ax3 = fig.add_subplot(1, 3, 3)
 
# 기본값
sns.countplot(x='class', palette='Set1', data=titanic, ax=ax1) 

# hue 옵션에 'who' 추가 
sns.countplot(x='class', hue='who', palette='Set2', data=titanic, ax=ax2) 

# dodge=False 옵션 추가 (축 방향으로 분리하지 않고 누적 그래프 출력)
sns.countplot(x='class', hue='who', palette='Set3', dodge=False, 
              data=titanic, ax=ax3)       

# 차트 제목 표시
ax1.set_title('titanic class')
ax2.set_title('titanic class - who')
ax3.set_title('titanic class - who(stacked)')

plt.show()


# In[29]:


# seaborn box-violin
# 그래프 객체 생성 (figure에 4개의 서브 플롯을 생성)
fig = plt.figure(figsize=(15, 10))   
ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 2)
ax3 = fig.add_subplot(2, 2, 3)
ax4 = fig.add_subplot(2, 2, 4)
 
# 박스 그래프 - 기본값
sns.boxplot(x='alive', y='age', data=titanic, ax=ax1) 

# 바이올린 그래프 - hue 변수 추가
sns.boxplot(x='alive', y='age', hue='sex', data=titanic, ax=ax2) 

# 박스 그래프 - 기본값
sns.violinplot(x='alive', y='age', data=titanic, ax=ax3) 

# 바이올린 그래프 - hue 변수 추가
sns.violinplot(x='alive', y='age', hue='sex', data=titanic, ax=ax4) 

plt.show()


# In[30]:


# seaborn joint
# 조인트 그래프 - 산점도(기본값)
j1 = sns.jointplot(x='fare', y='age', data=titanic) 

# 조인트 그래프 - 회귀선
j2 = sns.jointplot(x='fare', y='age', kind='reg', data=titanic) 

# 조인트 그래프 - 육각 그래프
j3 = sns.jointplot(x='fare', y='age', kind='hex', data=titanic) 

# 조인트 그래프 - 커럴 밀집 그래프
j4 = sns.jointplot(x='fare', y='age', kind='kde', data=titanic) 

# 차트 제목 표시
j1.fig.suptitle('titanic fare - scatter', size=15)
j2.fig.suptitle('titanic fare - reg', size=15)
j3.fig.suptitle('titanic fare - hex', size=15)
j4.fig.suptitle('titanic fare - kde', size=15)

plt.show()


# In[31]:


# seaborn facegrid
# 조건에 따라 그리드 나누기
g = sns.FacetGrid(data=titanic, col='who', row='survived') 

# 그래프 적용하기
g = g.map(plt.hist, 'age')


# In[40]:


# seaborn pairplot
# titanic 데이터셋 중에서 분석 데이터 선택하기 (숫자 데이터만 적용됨.)
titanic_pair = titanic[['age','pclass', 'fare']]

# 조건에 따라 그리드 나누기
g = sns.pairplot(titanic_pair)


# In[45]:


import pandas as pd
import folium


# In[ ]:


# 서울 지도 만들기 ( 스타일 변경 )
seoul_map2 = folium.Map(location=[37.55,126.98], tiles='Stamen Terrain', 
                        zoom_start=12)
seoul_map3 = folium.Map(location=[37.55,126.98], tiles='Stamen Toner', 
                        zoom_start=15)

# 지도를 HTML 파일로 저장하기
seoul_map2.save('./seoul2.html')
seoul_map3.save('./seoul3.html')


# In[72]:


# 서울 지도 만들기
seoul_map = folium.Map(location=[37.55,126.98], tiles='Stamen Terrain', 
                        zoom_start=12)
# 대학교 위치정보를 Marker로 표시
df = pd.read_excel('./test_data/서울지역 대학교 위치.xlsx')
df.rename(columns = {'Unnamed: 0' : 'name'},inplace=True)
df.set_index('name',inplace=True)
for name, lat, lng in zip(df.index, df.위도, df.경도):
    folium.Marker([lat, lng], popup=name).add_to(seoul_map)
# 지도를 HTML 파일로 저장하기
seoul_map.save('./seoul_colleges.html')


# In[75]:


# 대학교 위치정보를 CircleMarker로 표시
for name, lat, lng in zip(df.index, df.위도, df.경도):
    folium.CircleMarker([lat, lng],
                        radius=10,         # 원의 반지름
                        color='brown',         # 원의 둘레 색상
                        fill=True,
                        fill_color='coral',    # 원을 채우는 색
                        fill_opacity=0.7, # 투명도    
                        popup=name
    ).add_to(seoul_map)


# In[415]:


# 데이터 체크
df = sns.load_dataset('titanic')
df.info()
print(df.deck.unique()) # 컬럼의 고유 값 조회
print(df.deck.value_counts(dropna=False)) # 컬럼의 고유 값의 개수 조회
df.isnull() # 누락이면 True
df.notnull() # 누락이면 False
df.isnull().sum(axis=0) # 컬럼별 누락 값 조회


# In[135]:


# 1. 누락데이터 처리 - 제거하는 법
df_age = df.dropna(subset=['age'], how='any',axis=0)
# how = 'any' 옵션은 subset 일부 컬럼만 null 이어도 제거하는 것이다.
# how = 'all' 옵션은 subset 모든 컬럼이 null 이어야 제거하는 것이다.
print(len(df),len(df_age))

    # 누락데이터가 500개 이상인 컬럼을 삭제
df_t = df.dropna(axis=1, thresh= 500)
a = list(df.columns)
for i in df_t.columns:
    a.remove(i)
print(a) # deck 컬럼이 삭제되었음.

# 2. 누락데이터 처리 - 치환하는 법(평균)
mean_age = df['age'].fillna(df['age'].mean(axis=0))
print(df.age.loc[5],mean_age.loc[5])

    #np.floor : 소수점을 버리는 함수
mean_age = df['age'].fillna(np.floor(df['age'].mean(axis=0)))
print(df.age.loc[5],mean_age.loc[5])

# 3. 누락데이터 처리  - 문자의 경우
df['embark_town'].isnull().sum(axis=0)

    # 3-1. 가장 많이 나오는 값으로 치환
most_freq = df['embark_town'].value_counts(dropna=True).idxmax()
        # idxmax or idxmin = 최대, 최소 값을 가지는 인덱스를 출력한다.
df_em = df['embark_town'].fillna(most_freq)
print(df_em.unique(),most_freq)
    # 3-2. 앞 또는 뒤에 오는 값으로 치환
df['embark_town'].fillna(method='ffill')
df['embark_town'].fillna(method='bfill')
        # method = ffill / bfill = 앞부분 치환 / 뒷부분 치환


# In[175]:


df.info()


# In[192]:


# titanic 데이터를 로드해서 df에 저장한 후
df = sns.load_dataset('titanic')

# 1. 각 자료의 Nan의 개수를 확인
df.isnull().sum(axis=0)

# 2. Nan의 개수가 전체 데이터의 반절이 넘으면 컬럼 삭제
df.dropna(axis=1, thresh= len(df)/2 ,inplace=True)

for i in df.columns:
# 3. 컬럼의 데이터가 숫자인 경우, Nan을 해당 컬럼의 최소값으로 치환
    if df[i].dtypes in ['float64','int64'] :
        df[i].fillna(df[i].min(),inplace=True)

# 4. 컬럼의 데이터가 문자인 경우, Nan을 해당 컬럼에서 최대빈값으로 치환
    elif df[i].dtypes == 'object':
        df[i].fillna(df[i].value_counts(dropna=True).idxmax(),inplace=True)
    else:
        pass
df.isnull().sum(axis=0)


# In[327]:


# mpg dataset -> Nan 확인 후 평균값으로 결정 처리
df_mpg = sns.load_dataset('mpg')
df_mpg['horsepower'].fillna(df_mpg['horsepower'].mean(),inplace=True)
df_mpg.isnull().sum(axis=0)


# In[212]:


# 4. 중복데이터 처리 - 제거
df = pd.DataFrame({'c1' : ['a','a','b','a','a'],
                'c2' : [1, 1, 1, 2, 2],
                'c3' : [1, 1, 2, 2, 3]})
print(df)
df.duplicated() # 인덱스별 중복값 확인
df['c1'].duplicated() # True or False

# 중복 제거
df2 = df.copy()
df2 = df2.drop_duplicates()
print(df2)
    # 모든 컬럼이 중복인 것만 삭제
    
# 컬럼 기준으로 중복 제거
df3 = df.copy()
df3 = df3.drop_duplicates(subset=['c1','c2'])
    # 선택한 컬럼이 동시에 중복인 것을 제거하고 첫번째를 남김
print(df3)

df4 = df.copy()
df4 = df4.drop_duplicates(subset=['c1','c2'],keep='last')
    # 선택한 컬럼이 동시에 중복인 것을 제거하고 마지막을 남김
print(df4)


# In[420]:


# 5. 데이터 표준화
df_mpg = sns.load_dataset('mpg')
print(df_mpg.dtypes)

    # 5-1 단위 환산 : 갤런 -> 리터, 마일 -> 킬로미터로
mpg_to_kl = 1.60934/3.78541
df_mpg['kl'] = df_mpg['mpg']*mpg_to_kl
df_mpg['kl'] = df_mpg['kl'].round(2) # .round('소수점개수') : 반올림 함수
print(df_mpg.head(3))

    # 5-2 자료형 변환 (object -> category)
df_mpg['origin'] = df_mpg['origin'].astype('category')
df_mpg['origin'].dtypes
        # .astype('data type') : 데이터 자료형을 변환함.


# In[257]:


# 데이터 변환 예제
df_1 = pd.read_csv('./test_data/auto-mpg.csv')
df_mpg = sns.load_dataset('mpg')
df_1.columns = df_mpg.columns

df_1['horsepower'] = pd.to_numeric(df_1['horsepower'], errors='coerce')
# 해당 컬럼의 데이터를 모두 숫자로 변환하고, 숫자가 아닐경우엔 Nan으로 변환함.


# In[269]:


df_mpg = sns.load_dataset('mpg')
df_mpg['model_year'] = df_mpg['model_year'].astype('category')
df_mpg.info()


# 데이터 전처리 과정
# 1. 결측치 처리
#     .dropna(subset = [] , axis = 0 or 1)
#     .fillna()
#     .fillna(method='ffill')
#     .fillna(method='bfill')
#     pd.to_numeric(df[], errors='coerce')
# 
# 2. 테이터타입 변경(필요시)
#     .astype('data type')
# 
# 3. 단위 환산(필요시)
# 
# 4. 연속적인 데이터를 범주형 오브젝트로 변환
#     구간분할, np.histogram(), pd.cut()
#     
# 5. 원핫인코딩 작업
#     pd.get_dummies, sklearn.preprocessing.LabelEncoder() or OneHotEncoder()
#     
# 6. 정규화 : 숫자데이터를 0 ~ 1사이의 수로 변환하는 작업
#     모든 값을 최댓값으로 나눈다.
# 
# Now : 머신러닝을 위하여, 연속형 자료를 범주형으로 바꾸면서, 값을 숫자로 변환하는 과정
# 

# In[336]:


count, a =  np.histogram(df_mpg['horsepower'], bins=3)
print(a)


# In[427]:


# 연속형 데이터를 범주형 데이터로 새로 만들기
# 마력을 상,중,하로 나누어서 범주형 데이터로 만드는 예제.
df_mpg.dropna(subset = ['horsepower'] , axis=0, inplace=True) # 공백제거

# np.histogram 함수로 연속형 데이터를 상,중,하 범주로 나눌 기준을 만든다.
print(np.histogram(df_mpg['horsepower'], bins=3))
count, hp_bins_div = np.histogram(df_mpg['horsepower'], bins=3)

# 각 범주의 이름에 대한 리스트
bins_names = ['low power', 'middle power', 'high power'] 

# 새로운 범주형 데이터 컬럼을 만든다.
df_mpg['hp_bin'] = pd.cut(x=df_mpg['horsepower'],
                         bins = hp_bins_div, # 범주의 기준값
                         labels = bins_names, # 범주의 이름
                         include_lowest = True # 작은 경계값을 포함한다. 
                         )
df_mpg['hp_bin'].value_counts()


# In[424]:


pd.get_dummies(df_mpg['hp_bin']) # 범주데이터가 컬럼명으로 나누고, 값은 0 1로 만듬


# In[428]:


# sklearn 패키지의 preprocessing 모듈의 원-핫-인코딩
from sklearn import preprocessing

# 전처리를 위한 encoder 객체 생성
label_encoder = preprocessing.LabelEncoder()
onehot_encoder = preprocessing.OneHotEncoder()

# label_encoder로 문자열 번주를 숫자형 범주로 변환
onehot_labeled = label_encoder.fit_transform(df_mpg['hp_bin'].tail(5))
onehot_labeled

# 2차원 배열
onehot_reshaped = onehot_labeled.reshape(len(onehot_labeled),1)
onehot_reshaped

# 희소행렬로 변환
onehot_fitted = onehot_encoder.fit_transform(onehot_reshaped)
print(type(onehot_fitted))
print(onehot_fitted)


# In[328]:


# 정규화 작업.
print(df_mpg.horsepower.describe())
print('\n')

df_mpg['horsepower'] = df_mpg['horsepower'] / abs(df_mpg['horsepower'].max())
# 해당 컬럼의 모든 값을 최댓값으로 나누어서 저장함.
# 모든 값은 0 ~ 1사이의 값으로 변함 (정규화)

print(df_mpg.horsepower.describe())
print('\n')


# In[349]:


# 시계열 데이터
df = pd.read_csv('./test_data/stock-data.csv')
df.info()

# Data의 자료타입이 오브젝트 -> datetime 형식으로 변환
df['New_date'] = pd.to_datetime(df['Date'])
df.set_index('New_date',inplace=True) # 인덱스로 지정
df.drop('Date',axis=1, inplace=True) # 기존 데이트는 삭제함.
df.reset_index(inplace=True)
df['New_date'].dt.year # 연도
df['New_date'].dt.month # 월
df['New_date'].dt.day # 일


# In[359]:


df.head()


# In[375]:


# stock-data 를 파일에서 읽어 DataFrame으로 저장
df = pd.read_csv('./test_data/stock-data.csv')
# Date 컬럼을 DateTime으로 변경
df['Date'] = pd.to_datetime(df['Date'])
# Date 컬럼을 인덱스로 설정
df.set_index('Date',inplace=True)
# 날짜별로 start와 close 값을 그래프로 나타내보세요.
fig = plt.figure()
ax = fig.add_subplot(1,1,1)

ax.plot(df['Close'])
ax.plot(df['Start'])

plt.show()


# In[387]:


import matplotlib.cbook as cbook

# stock-data 를 파일에서 읽어 DataFrame으로 저장
df = pd.read_csv('./test_data/stock-data.csv')
# Date 컬럼을 DateTime으로 변경
df['Date'] = pd.to_datetime(df['Date'])
# Date 컬럼을 인덱스로 설정
df.set_index('Date')

# 날짜별로 start와 close 값을 그래프로 나타내보세요.
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
ax1.plot(df.Date, df.Start, 'o-')
ax1.plot(df.Date, df.Close, 'o-')
ax1.set_title("Stock plot")

plt.show()


# In[408]:


import numpy as np
import pandas as pd
import seaborn as sns
sns.set_theme(style="whitegrid")

# stock-data 를 파일에서 읽어 DataFrame으로 저장
df = pd.read_csv('./test_data/stock-data.csv')
# Date 컬럼을 DateTime으로 변경
df['Date'] = pd.to_datetime(df['Date'])
# Date 컬럼을 인덱스로 설정
df.set_index('Date',inplace=True)
data = df.loc[ : ,'Close' : 'Low']

sns.lineplot(data=data, palette="tab10", linewidth=2.5)


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook

# Load a numpy record array from yahoo csv data with fields date, open, close,
# volume, adj_close from the mpl-data/example directory. The record array
# stores the date as an np.datetime64 with a day unit ('D') in the date column.
r = (cbook.get_sample_data('goog.npz', np_load=True)['price_data']
     .view(np.recarray))
r = r[-30:]  # get the last 30 days

# first we'll do it the default way, with gaps on weekends
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4))
ax1.plot(r.date, r.adj_close, 'o-')
ax1.set_title("Default")
fig.autofmt_xdate()

# next we'll write a custom formatter
N = len(r)
ind = np.arange(N)  # the evenly spaced plot indices


def format_date(x, pos=None):
    thisind = np.clip(int(x + 0.5), 0, N - 1)
    return r.date[thisind].item().strftime('%Y-%m-%d')


ax2.plot(ind, r.adj_close, 'o-')
# Use automatic FuncFormatter creation
ax2.xaxis.set_major_formatter(format_date)
ax2.set_title("Custom tick formatter")
fig.autofmt_xdate()

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
