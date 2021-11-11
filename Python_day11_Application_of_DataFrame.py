#!/usr/bin/env python
# coding: utf-8

# In[2]:


# %load graphicsource_include
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


# In[ ]:





# In[186]:


# 시리즈 함수 매핑 : 시리즈.apply()
# 데이터프레임에 함수 매핑 : 데이터프레임.applymap()
# 데이터프레임 각 열에 함수 매핑 : 데이터프레임.apply(함수명)
# 데이터프레임 각 행에 함수 매핑 : 데이터프레임.apply(함수명, axis=1)

# 시리즈의 개별 원소에 함수 매핑 : .apply()
df = sns.load_dataset('titanic')
def add_10(n):
    return n+10
sr1 = df['age'].apply(add_10)

# 람다 함수도 적용가능함.
sr2 = df['age'].apply(lambda x : x+10)
sr3 = df['age'].apply(lambda x : add_10(x))
sr3.head()


# In[191]:


# 데이터프레임의 함수 매핑 : DataFrame.applymap()

df_1 = df[['age','fare']]

print(df_1.applymap(add_10).head())

# apply의 axis 연산
def max_min(x):
    return x.max() - x.min()
result = df_1.apply(lambda x : x.max() - x.min(), axis=1) # age - fare
result.head()


# In[192]:


# apply를 활용한 새로운 컬럼 생성
df_1['sum'] = df_1.apply(lambda x : x.sum(), axis=0) # age + fare
df_1.head()


# In[194]:


# 데이터프레임 함수매핑 : 데이터프레임.pipe(함수명) : 반환하는 종류에 따라 반환타입 정해짐.
df = sns.load_dataset('titanic')
df_1 = df[['age','fare']]

def missing_value(x):
    return x.isnull()

def missing_count(x):
    return missing_value(x).sum()

def total_number_missing(x):
    return missing_count(x).sum()

result_df = df_1.pipe(missing_value) #  -> True False
result_ser = df_1.pipe(missing_count) # -> Series
result_val = df_1.pipe(total_number_missing) # -> valuse
print(result_df.head())
print(result_ser)
print(result_val)


# In[59]:


# 컬럼 재구성 : 컬럼 순서 변경
# titanic 데이터프레임에서 'survived' 에서 'age'
df.head()

df_form = df.loc[0:4, 'survived' : 'age']
print(df_form)

# 컬럼을 index -> array -> list 로 변환한다.
columns_list = list(df_form.columns.values)
print(columns_list)
# 컬럼을 정렬하여 할 수있음.
df_form = df_form[sorted(columns_list)]
print(df_form)
# 반대로 정렬도 가능
df_form2 = df_form[sorted(columns_list,reverse=True)]
print(df_form2)


# In[209]:


# 컬럼 내 데이터를 나누어서 컬럼을 분리하기
# ex. 연월일 -> 연 / 월 / 일 컬럼으로 분리
stock = pd.read_excel('./test_data/stock.xlsx')
stock

stock['연월일'] = stock['연월일'].astype('str')
date_s = stock['연월일'].str.split('-') # 리스트로 데이터가 분리되어 반환됨.
print(dates.head()) # Series

stock['연'] = date_s.str.get(0)
stock['월'] = date_s.str.get(1)
stock['일'] = date_s.str.get(2)
stock.head()


# In[32]:


# 자료의 일부만 잘라냄
titanic = sns.load_dataset('titanic')
mask1 = (titanic.age >= 10) & (titanic.age < 20)

#df_teenage = titanic.loc[mask1, : ]
df_teenage = titanic.loc[(titanic.age >= 10) & ( titanic.age < 20), :]
df_teenage

# df_female -> 10대 미만이고 여성만 검색
df_female = titanic.loc[(titanic.age < 10) & (titanic.sex == 'female'), : ]
df_female

# df_1 : 10대 미만이거나 60대 이상인 데이터의 'age', 'sex', 'class' 만 검색
mask2 = (titanic.age < 10) | (titanic.age >= 60)
df_1 = titanic.loc[mask2,['age', 'sex', 'class']]
df_1

# 'sibsp' : 배우자의 수가 3 또는 4, 또는 5인 승객의 자료만 검색
mask3 = titanic.sibsp == 3
mask4 = titanic.sibsp == 4
mask5 = titanic.sibsp == 5
df_2 = titanic.loc[mask1 | mask2 | mask3, : ]
df_2

# 컬럼.isin ('추출값의 리스트')
mask6 = titanic['sibsp'].isin([3,4,5])
df_3 = titanic.loc[mask6, : ]
df_3

df_4 = titanic.loc[ mask6 & mask2, : ] # 여러개의 마스크 조건 활용
df_4.head()


# In[36]:


# 데이터프레임 합치기
df1 = pd.DataFrame({'a': ['a0', 'a1', 'a2', 'a3'],
                    'b': ['b0', 'b1', 'b2', 'b3'],
                    'c': ['c0', 'c1', 'c2', 'c3']},
                    index=[0, 1, 2, 3])
 
df2 = pd.DataFrame({'a': ['a2', 'a3', 'a4', 'a5'],
                    'b': ['b2', 'b3', 'b4', 'b5'],
                    'c': ['c2', 'c3', 'c4', 'c5'],
                    'd': ['d2', 'd3', 'd4', 'd5']},
                    index=[2, 3, 4, 5])
df3 = pd.concat([df1,df2])
df3 # 기존의 인덱스는 중복 상관없이 그대로 추가가 됨.
# 컬럼의 경우 중복이 있으면, 합쳐지게되며, 존재하지않는 값은 Null로 출력된다.


# In[37]:


df4 = pd.concat([df1,df2],ignore_index=True)
df4 # 기존 인덱스 상관없이 새로 인덱스를 부여한다.


# In[38]:


df5 = pd.concat([df1,df2], axis=1)
df5 # 컬럼을 중복상관없이 그대로 추가가 됨.
# 인덱스의 경우 중복이 있으면, 합쳐지게되며, 존재하지않는 값은 Null로 출력된다.


# In[213]:


# 데이터프레임 병합
df1 = pd.read_excel('./test_data/stock_price.xlsx')
df2 = pd.read_excel('./test_data/stock_valuation.xlsx')
df1.info()
df2.info()
# 두 개의 데이터프레임 병합 : id 컬럼으로 병합
df_merge = pd.merge(df1,df2) # 같은 컬럼명이 있으면 해당 컬럼의 값이 같은 자료를 병합
df_merge # 이 경우에는 id가 동일한 데이터만 병합되어 반환된다.


# In[214]:


pd.merge(df1,df2, how = 'outer').head() # outer는 합집합, inner는 교집합을 의미


# In[215]:


pd.merge(df1,df2, how = 'right').head() # 오른쪽 컬럼부터 일치하는 것을 기준. = name


# In[216]:


pd.merge(df1,df2, how = 'left').head() # 왼쪽 컬럼부터 일치하는 것을 기준. = stock_name


# In[211]:


pd.merge(df1,df2, left_on = 'stock_name', right_on = 'name')
# 왼쪽과 오른쪽 컬럼 모두 일치하는 것을 기준으로 함.


# In[54]:


pd.merge(df1[df1['price']<5000],df2)
# df1의 price가 5000미만인 자료만 merge


# In[210]:


# join 함수 활용 : 특정 컬럼을 인덱스로 지정하여 인덱스가 일치하는 것끼리 병합
df1 = pd.read_excel('./test_data/stock_price.xlsx',index_col = 'id')
df2 = pd.read_excel('./test_data/stock_valuation.xlsx',index_col = 'id')
df1.join(df2, how='right').head()


# In[66]:


df = sns.load_dataset('titanic')
df = df.loc[ : , ['age','sex','class','fare','survived']]
len(df)
df['class'].unique()

grouped = df.groupby('class') # 클래스가 곧 인덱스가 된다.
list(grouped)


# In[67]:


for key, group in grouped: # key = index
    print('key : {} - {}명'.format(key,len(group)))
    print(group.head())


# In[81]:


grouped.mean() # 그룹별 집계 함수의 값 : min, max, count, std, var

grouped.get_group('First').head(3) # 특정 그룹만 반환하기

# class 와 sex를 그룹으로 만들어서 각 그룹별 컬럼별 평균값을 구하세요,
group_1 = df.groupby(['class','sex'])
group_1.mean()

# 그룹의 fare의 표준편차
group_1['fare'].std()

# 그룹.agg(함수리스트) : 각각 함수 모두 계산함
group_1.agg([min,max])

# 그룹.agg({컬럼:함수, 컬럼2:함수2}) : 각 컬럼에 함수를 지정할 수도 있음.
group_1.agg({'age':[min,max],'fare': 'std', 'survived' : 'mean'})


# In[102]:


# mpg 데이터를 로드해서 origin로 그룹을 지어서
# mpg, weight, cylinders, horsepower의 그룹별 평균을 구하세요.
# 결측치 처리후에 그룹 연산

df_mpg = sns.load_dataset('mpg')[['mpg','weight','cylinders','origin','horsepower']]
df_mpg.isna().sum()
# horsepower에 공백이 6개 있음.
df_mpg.horsepower.fillna(df_mpg.horsepower.mean(),inplace=True)
# 공백을 평균값으로 처리
df_mpg.dropna(subset=['horsepower'],axis=0,inplace=True)
# 공백이 있을 경우, 제거

df_mpg_origin = df_mpg.groupby('origin')
df_mpg_origin.mean()


# In[217]:


df = titanic.loc[ : ,['age','sex','class','fare','survived']]
df.age.fillna(df.age.mean(),inplace=True)
grouped = df.groupby('class')
age_mean = grouped.age.mean()
age_std = grouped.age.std()

for key, group in grouped.age:
    group_zeros = (group - age_mean.loc[key]) / age_std.loc[key]
    print("key :", key)
    print(group_zeros.head(3))
# 클래스별 (나이 - 나이의 평균) / 나이 표준편차를 구했음.

def z_score(x):
    return (x - x.mean())/x.std()
        
age_zeros = grouped.age.transform(z_score) # 각 데이터에 함수를 적용
age_zeros.head(3)


# In[152]:


# 그룹에 대한 조건식
print(len(grouped.age))
grouped.filter(lambda x : len(x) >= 200)

# age의 평균이 30미만인 그룹만
grouped.filter(lambda x : x.age.mean() < 30)

# 그룹 객체에 함수 매핑
grouped.apply(lambda x : x.describe())

# 그룹 객체 age에 대하여 평균이 30미만
age_filter = grouped.apply(lambda x : x.age.mean() < 30)
print(age_filter)

for x in age_filter.index:
    if age_filter[x] == True: # 조건이 True인 그룹의 정보만 출력
        print('key :', x)
        print(grouped.get_group(x).head())


# In[206]:


# class와 sex를 그룹으로 생성
group_2 = df.groupby(['class','sex'])
gdf = group_2.mean()
gdf
gdf.loc["First"]
gdf.loc[('First','female')] # multi index 접근 방법
gdf.xs('female', level='sex') # sub index 접근
gdf.xs('First', level='class')


# In[219]:


# 피벗 테이블
pdf = pd.pivot_table(df, index = ['class','sex'], # 행 위치에 들어갈 컬럼
                     columns = 'survived', # 열 위치에 들어갈 컬럼
                     values = ['fare','age'], # 데이터로 사용할 값
                     aggfunc= ['max','min']) # 집계함수, defalt = mean
pdf


# In[218]:


# 피벗테이블 인덱싱
pdf.xs(['First'])
pdf.xs('male', level='sex')
pdf.xs('min', axis=1)
pdf.xs(('min','age'), axis=1)
pdf.unstack(['class','sex']) # 행 인덱스를 열 인덱스로 변경
pdf.stack('survived') # 열 인덱스는 행인덱스로 변경


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
