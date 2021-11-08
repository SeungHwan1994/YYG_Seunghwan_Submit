#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[27]:


list_data = [1, 3, 5, 7, 9]

s_data_l = pd.Series(list_data)

print(s_data_l)
print(s_data_l.index)
print(s_data_l.values)


# In[26]:


# series 생성 방법 : pd.Series(list) : index는 0부터 숫자 생성, 값은 원소로
dict_data = { 'a' : 1, 'b' : 2, 'c' : 3}

s_data = pd.Series(dict_data)

print(s_data)
print(s_data.index)
print(s_data.values)


# In[18]:


# series 접근 방법 : 정수형 인덱스, 인덱스명으로 접근

print(s_data[0],s_data['a']) # 원소를 1개를 선택

print(s_data[[0,1]]) # [start : end ] -> start 포함, end 포함X
print(s_data[0:2])

print(s_data['a':'c']) # 인덱스명으로 접근 [ start : end ] -> 둘다 포함
print(s_data[['a','c']]) # 여러개를 출력하고자 하면 [ 인덱스명 또는 인덱스 ]


# In[29]:


# 튜플 또는 리스트에 인덱스명을 따로 부여하는 방법
# 검색을 하거나 값을 입력하거나 -> 여러 개의 값을 처리 : 리스트 [ , , , ...]

tup_data = (1, 2, 3)
s_data = pd.Series(tup_data, index=['a', 'b', 'c'])
print(s_data)

list_data = [ 1, 2, 3]
index_data = ['a', 'b', 'c']
s_data = pd.Series(list_data, index=index_data)
print(s_data)


# In[53]:


# 데이터프레임 : 시리즈가 모여서 2차원 배열이 됨.
# pd.DataFrame(2차원 데이터, index = [], columns= [])

dict_data = { 'c0' : [1,2,3], 'c1' : [4,5,6], 'c3' : [7,8,9]}
df = pd.DataFrame(dict_data)
print(df)

# 데이터 프레임의 인덱스 / 열 이름 설정

df = pd.DataFrame(dict_data, index=['a', 'b', 'c'], columns=['c0', 'c1', 'c3'])
print(df)
print(df.index)
print(df.columns)


# In[255]:


# 숫자는 0부터 10까지의 자료를 가지고 2차원 배열로 ( 5 * 2 )로 구성한 데이터프레임 생성
# columns : 'c_1', 'c_2' 로, index : 'idx1', 'idx2', 'idx3', 'idx4', 'idx5'
# 데이터 프레임의 변수를 df로, 자료의 인덱스명과 컬럼명을 출력, 전체자료 출력
list_data = np.arange(10).reshape(5,2) + 1 # 각 1차원 배열의 수에 1을 더함.

df = pd.DataFrame(list_data,index=['idx1', 'idx2', 'idx3', 'idx4', 'idx5'],
                 columns = ['c_1','c_2'])


print(df.index)
print(df.columns)
print(df)


# In[256]:


# 인덱스 변경
df.index = ['a','b','c','d','e']
df.columns = ['one','two']
# 기존 인덱스 및 칼럼 변경
df.rename(index = {'b': 'ch1'},inplace = True) 
# inplace = True를 해줘야 기존 이름 변경됨.
print(df)

#인덱스 삭제, 컬럼 삭제
df.drop('a',axis=0, inplace=True) # 인덱스 삭제 (행)
df.drop('one',axis=1, inplace=True) # 칼럼명 삭제 (열)
print(df)


# In[134]:


exam_data = {'수학' : [90,80,70],
             '영어' : [98,89,65],
             '음악' : [85,95,100],
             '체육' : [100,90,90]}

df = pd.DataFrame(exam_data, index = ['서준', '우현', '인아'])
df


# In[107]:


# 데이터프레임 df의 데이터를 복제하여 변수 df2에 저장
df = pd.DataFrame(exam_data, index = ['서준', '우현', '인아'])
df2 = df[ : ]
df2.drop('우현',inplace=True)
df


# In[108]:


# 데이터프레임 df의 데이터를 복제하여 변수 df2에 저장
df = pd.DataFrame(exam_data, index = ['서준', '우현', '인아'])
df2 = df.copy()
df2.drop('우현',inplace=True)
df


# In[109]:


# 만약 데이터를 복제하지않을 경우, 기존 데이터프레임까지 전부 바뀐다.
df = pd.DataFrame(exam_data, index = ['서준', '우현', '인아'])
df2 = df
df2.drop('우현',inplace=True)
df


# In[128]:


df = pd.DataFrame(exam_data, index = ['서준', '우현', '인아'])
print(df.loc['서준'])
df.loc[['우현','인아']]


# In[131]:


df4 = df.copy()
df4.drop(['영어','체육'], axis=1, inplace=True)
df4


# In[132]:


df5 = df.copy()
df5.drop(['서준','인아'], axis=0, inplace=True)
df5


# In[135]:


df.describe()


# In[138]:


# 데이터프레임 접근 방법, 행을 접근 : loc['index name'], iloc['numeric index']
# [ : ], [ [ , , , ...]]
# "서준"의 자료 검색
a = df.loc['서준']
print(type(a),a)
df.iloc[0]


# In[139]:


df.loc[['서준','인아']]


# In[140]:


df.loc['서준':'인아']


# In[143]:


df.iloc[0:2]


# In[145]:


df.수학


# In[162]:


df[['수학','체육']]


# In[163]:


df[['수학' , '음악']]


# In[174]:


df['영어' : '수학'] # 안되는 것 같은데..?? 열이름은 순서가 없다. : 사용이 불가능함.


# In[ ]:


# 데이터프레임을 접근하는 방법
# 인덱스로 접근하는 방법 이름과 정수 인덱스
# 이름으로 접근 df.loc[인덱스명] : 범위를 정할 때는 시작 : 끝, 여러 개의 값 [ , ...]
# 정수 인덱스로 접근 df.iloc


# In[179]:


# 특정 원소에 접근, 행과열을 이용
print(df.loc['서준','음악'])
print(df.iloc[0,2])


# In[185]:


# 여러개 선택도 가능
print(df.loc['서준',['수학','영어']])
print(df.iloc[0,[0,1]])
print(df.loc['우현','영어' : '체육'])
print(df.iloc[1,1 : 4])


# In[207]:


df = pd.DataFrame(exam_data, index = ['서준', '우현', '인아'])
# 서준과 우현의 영어와 수학 점수 출력 (인덱스명)
print('서준과 우현의 영어와 수학 점수 출력\n',
      df.loc[['서준','우현'],['영어','수학']])
# 1행부터 끝까지 영어 점수만 출력(정수인덱스)
print('1행부터 끝까지 영어 점수만 출력\n',
      df.iloc[ : , 1])
# 마지막 자료의 마지막 과목만 출력 (정수인덱스)
print('마지막 자료의 마지막 과목만 출력\n',
      df.iloc[-1,-1])
# 우현의 모든 점수 출력 [ 인덱스 명으로]
print('우현의 모든 점수 출력\n',
      df.loc['우현', : ])
#우현과 인아의 모든 점수 출력 (정수 인덱스로)
print('우현과 인아의 모든 점수 출력\n',
      df.iloc[1 : 3, : ])


# In[238]:


# 열데이터 추가
df = pd.DataFrame(exam_data, index = ['서준', '우현', '인아'])
df['국어'] = [90,90,90]
df['영어2'] = df['영어']
df


# In[216]:


# 행데이터 추가
df = pd.DataFrame(exam_data, index = ['서준', '우현', '인아'])
df.loc['철수'] = [80,70,90,100]
df.loc['서준2'] = df.iloc[0]
df


# In[242]:


# 데이터프레임 제거
df.loc['서준2'] = df.iloc[0]
df.drop('서준2',axis = 0, inplace = True)
df['영어2'] = df['영어']
df.drop('영어2',axis = 1, inplace = True)
df


# In[250]:


df.transpose() # 행과 열을 서로 바꿈
df.T


# In[4]:


# 데이터프레임 : 인덱스 활용
exam_data = {'이름' : ['서준', '우현', '인아'],
             '수학' : [90, 80, 70],
             '영어' : [98, 89, 95],
             '음악' : [85,95,100],
             '체육' : [100,90,90]}

df = pd.DataFrame(exam_data)

df.set_index('이름',inplace=True)
df


# In[28]:


import pandas as pd
import numpy as np

dict_data = {'c1' : [1,2,3],'c2' : [4,5,6],'c3' : [7,8,9],
              'c4' : [10,11,12], 'c5' : [13,14,15]}
# reindex 는 기존 인덱슨느 그대로 사용해야지 기존 값이 변화하지 않는다.
df = pd.DataFrame(dict_data, index = ['r0', 'r1', 'r2'])
new_index = ['r0', 'r1', 'r2', 'r3', 'r4']
df = df.reindex(new_index, fill_value= 0 )
df


# In[32]:


# 인덱스 리셋 -> 인덱스가 컬럼으로 변경되고 그 자리에 정수형 인덱스가 들어온다,
df.reset_index()
df

# 인덱스를 설정 df.set_index(컬럼명), *inplace = True 기존의 인덱스 변경
# 인덱스 재배치 df.reindex(인덱스배열),
# 인덱스 제거 df.reset_index()


# In[36]:


# 컬럼 기준 정렬
df.sort_values(by = 'c1',ascending=False)


# In[39]:


# 인덱스 기준 정렬
df.sort_index(ascending=False)

# 이름을 인덱스로 설정
# 영어 점수로 정렬해서 출력
# 인덱스로 정렬해서 출력


# In[147]:


# 이름을 인덱스로 설정
# 영어 점수로 정렬해서 출력
# 인덱스로 정렬해서 출력

exam_data = {'이름' : ['서준', '우현', '인아'],
             '수학' : [90, 80, 70],
             '영어' : [98, 89, 95],
             '음악' : [85,95,100],
             '체육' : [100,90,90]}

test_df = pd.DataFrame(exam_data)

test_df.set_index("이름",inplace = True)

print(test_df.sort_values('영어'))
print(test_df.sort_index())

# 인덱스제거
test_df.reset_index( inplace = True)
print(test_df)


# In[157]:


# Series 연산
# dictonary로 판다스 시리즈 만들기

student1 = pd.Series({'국어' : 100, '영어' : 80, '수학' : 90, '과학' : 100})
student2 = pd.Series({'수학' : 80, ' 국어' : 90, '영어' : 80})
# student1 + student2

ss_add = student1.add(student2, fill_value = 0)
ss_sub = student1.sub(student2, fill_value = 0)
ss_mul = student1.mul(student2, fill_value = 0)
ss_div = student1.div(student2, fill_value = 0)

# 시리즈의 계산결과를 데이터프레임으로 합치기

results = pd.DataFrame([ss_add,ss_sub,ss_mul,ss_div],
                       index=['add','sub','mul','div'])
results


# In[163]:


# seaborn 모듈에서 데이터를 불러옴
import seaborn as sns

titanic = sns.load_dataset('titanic')
titanic.head() # 처음 5라인을 보여줌


# In[168]:


# age와 fare 컬럼만 가져오기
titanic_agefare = titanic[['age','fare']]
titanic_agefare.head()


# In[172]:


sum_titanic_agefare = titanic_agefare + 10
sum_titanic_agefare.tail()


# In[173]:


titanic_agefare.add(10,fill_value=0).tail()


# In[186]:


titanic.head() # 데이터셋 5개까지만 표현


# In[183]:


titanic.info() # 데이터프레임 컬럼과 데이터 갯수등의 정보를 보여줌


# In[185]:


titanic_agefare.describe() # 데이터셋의 통계치를 보여줌.


# In[188]:


# 외부파일 다루기
# read_csv_sample.csv 작업하고 있는 디렉토리에 저장.

import os
os.getcwd
os.listdir('./test_data')


# In[190]:


# csv 파일 데이터 가져오기
file_name = "./test_data/read_csv_sample.csv"
df = pd.read_csv(file_name) # header 옵션이 없으므로, 첫 행이 컬럼이 된다.
df


# In[260]:


# 첫 라인을 데이터로 인식
df = pd.read_csv(file_name,header = None) # header가 없는 상태로 읽음
print(df,'\n\n\n')
#인덱스 컬럼을 지정하면서 데이터를 가져옴
df2 = pd.read_csv(file_name,index_col = 'c0')
#인덱스 컬럼과 컬럼명을 지정하면서 데이터 가져옴
df3 = pd.read_csv(file_name, index_col = 'a',skiprows = 1,
                   names = ['a','b','c','d'])
#csv 파일로 저장
df3.to_csv("./test_Data/to_csv_sample.csv")
print(df3)


# In[244]:


# seaborn의 titanic 자료의 ages, fare, class, 컬럼만추출, 10번째 데이터 부터 60번째
#데이터까지만 추출함.
#위 데이터를 ./test_data/titanic_sample.csv 파일로 저장

import seaborn as sns

titanic = sns.load_dataset('titanic')

titanic_datasample = titanic[['age','fare','class']]
titanic_datasample = titanic_datasample.iloc[10 : 60 + 1, : ]

titanic_datasample.to_csv('./test_data/titanic_sample.csv')


# In[243]:


# 임의로 20개의 데이터만 추출, random_state = 일정한 값 가져오기
titanic_datasample = titanic.sample(20, random_state = 15)
titanic_datasample


# In[250]:


# excel 파일 read vs write
df1 = pd.read_excel('./test_data/남북한발전전력량.xlsx') # header = 0 default 옵션
df2 = pd.read_excel('./test_data/남북한발전전력량.xlsx',header = None)

df1.to_excel('./test_data/sample_to_excel.xlsx')
df1


# In[253]:


# 파일을 형식에 따라 read write
# csv,excel,json,html

df4 = pd.read_json('./test_data/read_json_sample.json')
df4.to_json('./test_data/to_json_sample.json')
df4


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
