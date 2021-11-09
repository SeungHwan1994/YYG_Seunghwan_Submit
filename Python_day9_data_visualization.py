#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import seaborn as sns

df = sns.load_dataset('titanic')


# male의 생존자 수, 사망자수
m_live = df[df['sex'] == 'male' and df['alive'] == 'yes']['alive'].count()
m_dead = df[df['sex'] == 'male' and df['alive'] == 'no']['alive'].count()
# female의 생존자수, 사망자수
f_live = df[df['sex'] == 'female' and df['alive'] == 'yes']['alive'].count()
f_dead = df[df['sex'] == 'female' and df['alive'] == 'no']['alive'].count()
#위의 데이터를 데이터프레임으로 구성
df_new = pd.DataFrame([m_live,m_dead,f_live,f_dead],index = ['m_live','m_dead','f_live','f_dead'])
df_new


# In[ ]:


# 시리즈, 데이터프레임 ( 2차원 배열, 인덱스, 컬럼)
# df.loc[인덱스명, 컬럼명], df.iloc[인덱스숫자, 컬럼 숫자]
# df.loc[ [ , , , ...], [ , , , ...] ]
# df.loc[ start : end , start : end ]


# In[2]:


# titanic에서 성별로 생존자수와 사망자수를 데이터프레임으로 작성해보세요.
import seaborn as sns
import pandas as pd
import numpy as np

df = sns.load_dataset('titanic')


# In[16]:


df.head()
df_sex = df[['sex' , 'alive']]
df_sex.head()

male_y = df_sex[ (df['sex'] == 'male') & (df['alive'] == 'yes') ]['sex'].count()
male_n = df_sex[ (df['sex'] == 'male') & (df['alive'] == 'no') ]['sex'].count()

female_y = df_sex[ (df['sex'] == 'female') & (df['alive'] == 'yes') ]['sex'].count()
female_n = df_sex[ (df['sex'] == 'female') & (df['alive'] == 'no') ]['sex'].count()

df_sex = pd.DataFrame({
    'male' : [male_y, male_n],
    'female' : [female_y,female_n]},
    index = ['생존자수','사망자수'])

df_sex


# In[33]:


# unique() = 값의 종류를 확인, 중복제거


df_class = df[['class','alive']]

class_cnt = [[df_class[('First' == df_class['class']) & 
                       (df_class['alive'] == 'yes')]['class'].count(),
              df_class[('First' == df_class['class']) & 
                       (df_class['alive'] == 'no')]['class'].count()],
             [df_class[('Second' == df_class['class']) &
                (df_class['alive'] == 'yes')]['class'].count(),
              df_class[('Second' == df_class['class']) &
                (df_class['alive'] == 'no')]['class'].count()],
             [df_class[('Third' == df_class['class']) &
                (df_class['alive'] == 'yes')]['class'].count(),
             df_class[('Third' == df_class['class']) &
                (df_class['alive'] == 'no')]['class'].count()]]
             
class_cnt

columns_name = ['생존자 수','사망자 수']
index_name = ['First','Second','Third']

df_class_live = pd.DataFrame(class_cnt, index = index_name,
                            columns = columns_name)

df_class_live.T


# In[42]:


# numpy 이용

list_class_cnt = [df_class[('First' == df_class['class']) & 
                       (df_class['alive'] == 'yes')]['class'].count(),
              df_class[('First' == df_class['class']) & 
                       (df_class['alive'] == 'no')]['class'].count(),
             df_class[('Second' == df_class['class']) &
                (df_class['alive'] == 'yes')]['class'].count(),
              df_class[('Second' == df_class['class']) &
                (df_class['alive'] == 'no')]['class'].count(),
             df_class[('Third' == df_class['class']) &
                (df_class['alive'] == 'yes')]['class'].count(),
             df_class[('Third' == df_class['class']) &
                (df_class['alive'] == 'no')]['class'].count()]
print(list_class_cnt)
array_class_cnt = np.array(list_class_cnt)
array_class_cnt = array_class_cnt.reshape(3,2)

df_class_cnt =  pd.DataFrame(array_class_cnt, index = index_name,
                            columns = columns_name)
df_class_cnt.T


# In[44]:


# unique() 함수의 활용
class_cnt = {}
for class_n in df['class'].unique():
    class_value = [ df[(df['class'] == class_n ) &
                       (df['alive'] == 'yes')]['class'].count(),
                   df[(df['class'] == class_n ) &
                       (df['alive'] == 'no')]['class'].count()]
    class_cnt[class_n] = class_value
print(class_cnt)


# In[48]:


#html 데이터 가져오기
html = pd.read_html('./test_data/sample.html')

len(html)
for i in range(len(html)):
    print(html[i])
    
df = html[1]
df.set_index('name', inplace = True)
df


# In[50]:


# pandas.ExcelWriter("파일명") 여러개의 데이터프레임을 하나의 파일로 저장가능

data1 = {'name' : [ 'Jerry', 'Riah', 'Paul'],
         'algol' : [ "A", "A+", "B"],
         'basic' : [ "C", "B", "B+"],
          'c++' : [ "B+", "C", "C+"]}

data2 = {'c0':[1,2,3], 
         'c1':[4,5,6], 
         'c2':[7,8,9], 
         'c3':[10,11,12], 
         'c4':[13,14,15]}

df1 = pd.DataFrame(data1)
df1.set_index('name',inplace = True)

df2 = pd.DataFrame(data2)
df2.set_index('c0',inplace = True)

# 엑셀에 여러개의 데이터프레임을 각각의 sheet로 저장하는 방법.

write_file = pd.ExcelWriter("./test_data/excelsample.xlsx")
df1.to_excel(write_file, sheet_name = 'sheet1')
df2.to_excel(write_file, sheet_name = 'sheet2')
write_file.save()


# In[73]:


# 데이터셋을 가져올때, 데이터를 확인했을때 header를 넣을지 안넣을지 판단해야한다.
df = pd.read_csv("./test_data/auto-mpg.csv",header = None)
df_columns = ['mpg','cylinders','displacement','horsepower','weight',
                    'acceleration','modeel year','origin','name']
df.columns = df_columns

df.head() # 처음 5행
df.tail() # 마지막 5행
df.shape # 데이터의 행과 열의 개수
df.info() # 데이터프레임의 데이터속성 정보
df.dtypes # 컬럼들의 데이터타입ㅐ
df.mpg.dtypes # 특정 컬럼의 데이터타입
df.describe() # 데이터의 통계 정보
df.mpg.describe() # 데이터의 통계 정보
df.count() # 컬럼들의 데이터 갯수
df.mpg.count() # 특정 컬럼의 데이터갯수
df.describe(include = 'all') # 기술 통계의 모든 정보 확인
df.origin.value_counts() # 컬럼의 고유한 값의 갯수


# In[80]:


# 통계 함수의 적용
# 상관계수 df[[열이름 리스트]].corr()
df[df_columns].corr()

# 평균값 = mean()
print(df[['mpg','weight']].mean())
# 중간값 = median()
print(df['mpg'].median())
# min, max
print(df['mpg'].max())
print(df['mpg'].min())
# std = 표준편차
print(df['mpg'].std())


# In[94]:


# 판다스의 내장 그래프 plot()

df = pd.read_excel('./test_data/남북한발전전력량.xlsx')
df.head(8) # 0행 과 5행에 있는 정보만 가져옴
df_ns = df.iloc[[0,5], 3: ] # 1991년도 부터 가져옴
df_ns

#인덱스 명 부여
df_ns.index = ["South",'North']
df_ns.columns = df_ns.columns.map(int) #컬럼명의 데이터타입을 str -> int로 변경
df_ns.plot()
# 인덱스가 컬럼으로, 컬럼의 연도가 인덱스로 변형해서 그래프 확인
df_ns.T.plot() # x 축이 인덱스 / y 축은 값 / column을 그래프에 표시


# In[97]:


# 그래프의 종류 : hist, scatter, bar
df_ns.T.plot(kind = 'bar')
df_ns.T.plot(kind = 'barh')
df_ns.T.plot(kind = 'hist')


# In[102]:


# scatter plot, box plot 사용예시
df = pd.read_csv("./test_data/auto-mpg.csv",header = None)
df_columns = ['mpg','cylinders','displacement','horsepower','weight',
                    'acceleration','modeel year','origin','name']
df.columns = df_columns

df.plot(x="weight",y="mpg",kind='scatter')
df[['mpg','weight']].plot(kind='box')


# In[103]:


# 데이터분석 세팅
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

fig, ax = plt.subplots() # figure에 1개의 axe를 만듬
ax.plot([1,2,3,4],[1,4,2,3])


# In[113]:


df = pd.read_excel('./test_data/시도별 전출입 인구수.xlsx')
print(df['전출지별'].unique())
df.fillna(method = 'ffill',inplace = True)
#누락 데이터가 들어있는 행의 바로 앞에 위치한 행의 데이터값으로 채운다.
# 누락데이터(null or None) 처리가 가장 1순위로 진행한다.
print(df['전출지별'].unique())


# In[120]:


# 서울에서 다른 지역으로 이동한 자료만 추출
# 전출지가 서울특별시, 전입지는 서울특별시 이에 다른 지역
df_seoul = df[ (df['전출지별'] == '서울특별시') &
             (df['전입지별'] != '서울특별시')]
df_seoul
# 전출지별은 어차피 서울특별시이므로, 해당 컬럼을 삭제
df_seoul = df_seoul.drop('전출지별',axis = 1)
# 전입지별 컬럼을 인덱스로 지정
df_seoul.set_index('전입지별',inplace=True)
df_seoul


# In[192]:



# 한글 폰트 사용을 위해서 세팅
from matplotlib import font_manager, rc
font_path = "C:\Windows\Fonts\malgunbd.TTF"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)


# In[124]:


# 서울에서 경기도로 전입한 인구 데이터만 검색
seoul_one = df_seoul.loc['경기도']
type(seoul_one) # series

plt.plot(seoul_one.index, seoul_one.values)


# In[165]:


# fugyre size 조정
plt.figure(figsize =(14, 5))

plt.plot(seoul_one.index, seoul_one.values, marker='o', markersize=10)

# xlabel 과 ylabel을 정의
plt.title("Seoul to Gyungido", size = 30)
plt.xlabel("Period", size = 20)
plt.ylabel("Population", size = 20)

# x축을 로테이트
plt.xticks(size=10,rotation='vertical')
# 범례 추가
plt.legend(labels=['seoul -> Gyungi'],loc='best')
# 스타일 서식 지정
plt.style.use('ggplot')

# y축 범위 지정
plt.ylim(50000,800000)

# 주석 표시, 화살표
plt.annotate('',
            xy=(20,620000),
            xytext=(2,290000),
            xycoords='data',
            arrowprops=dict(arrowstyle="->",color='skyblue',lw=5),
            )

plt.annotate('',
            xy=(47,450000),
            xytext=(30,580000),
            xycoords='data',
            arrowprops=dict(arrowstyle="->",color='skyblue',lw=5),
            )
plt.show()


# In[162]:


# 한 화면에 여러개의 그래프 그리기
fig = plt.figure(figsize=(10,10))
ax1 = fig.add_subplot(2,1,1)
ax2 = fig.add_subplot(2,1,2)

ax1.plot(seoul_one,'o',markersize=10)
ax2.plot(seoul_one,marker='o',markerfacecolor='green',markersize=10,
         color='yellow',linewidth=2,label='Seoul -> Gyungi immigration')
ax2.legend(loc='best')

ax1.set_xticklabels(seoul_one.index,rotation='vertical')
ax2.set_xticklabels(seoul_one.index,rotation=75)

plt.show()


# In[185]:


# 서울에서 '충청남도','경상북도','강원도'로 이전한 자료만 선택, 하나의 plot에 여러개의 그래프 그림
col_years = list(map(str, range(1970,2018)))

df_3 = df_seoul.loc[['충청남도','경상북도','강원도'], col_years]
# df_3.head()

fig = plt.figure(figsize=(20,5))
ax = fig.add_subplot(1,1,1)

ax.plot(col_years, df_3.loc['충청남도', : ], label='Chungnam')  
ax.plot(col_years, df_3.loc['경상북도', : ], label='Gyungbook')
ax.plot(col_years, df_3.loc['강원도', : ], label='Gangwon')

ax.legend(loc='best')
ax.set_title('Seoul to')
ax.set_xlabel('Period')
ax.set_ylabel('Immigration No.')
ax.tick_params(axis='x', labelrotation=90)
plt.show()


# In[187]:


# 서울에서 충청남도, 경상북도, 강원도,'전라남도'로 이전한 자료만 선택
col_years = list(map(str,range(1970,2018)))
col_years

df_3 = df_seoul.loc[['충청남도','경상북도','강원도','전라남도'],col_years]

fig = plt.figure(figsize=(60,20))
ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(2,2,2)
ax3 = fig.add_subplot(2,2,3)
ax4 = fig.add_subplot(2,2,4)

ax1.plot(col_years, df_3.loc['충청남도', : ],label='Seoul to Chungnam')
ax2.plot(col_years, df_3.loc['경상북도', : ],label='Seoul to Gyungbook')
ax3.plot(col_years, df_3.loc['강원도', : ],label='Seoul to Gangwon')
ax4.plot(col_years, df_3.loc['전라남도', : ],label='Seoul to Jeonnam')

ax1.legend(loc='best')

ax1.set_title('Seoul to Chungnam, Gyungbook, Gangwon',size = 30)
ax1.set_xlabel('Period',size = 20)
ax1.set_ylabel('immigration',size = 20)
ax1.tick_params(axis='x',labelrotation=90)

ax2.legend(loc='best')

ax2.set_title('Seoul to Chungnam, Gyungbook, Gangwon',size = 30)
ax2.set_xlabel('Period',size = 20)
ax2.set_ylabel('immigration',size = 20)
ax2.tick_params(axis='x',labelrotation=90)

ax3.legend(loc='best')

ax3.set_title('Seoul to Chungnam, Gyungbook, Gangwon',size = 30)
ax3.set_xlabel('Period',size = 20)
ax3.set_ylabel('immigration',size = 20)
ax3.tick_params(axis='x',labelrotation=90)

ax4.legend(loc='best')

ax4.set_title('Seoul to Chungnam, Gyungbook, Gangwon',size = 30)
ax4.set_xlabel('Period',size = 20)
ax4.set_ylabel('immigration',size = 20)
ax4.tick_params(axis='x',labelrotation=90)
plt.show()


# In[224]:


# 영역 그래프로 표현
col_years = list(map(str, range(1970,2018)))

df_4 = df_seoul.loc[['충청남도','경상북도','강원도','전라남도'], col_years]
# df_3.head()
df_4 = df_4.T
df_4.index = df_4.index.map(int) # 연도를 전부 숫자로 변환함.

df_4.plot(kind='area',stacked = False, alpha=0.3, figsize = (10,5))
# area는 연도가 숫자로 되어있어야 계산이 가능함.

plt.title('Seoul to')
plt.xlabel('Period')
plt.ylabel('Immigration No.')

plt.show()


# In[223]:


# 막대그래프로 표현
col_years = list(map(str, range(2001,2018)))

df_4 = df_seoul.loc[['충청남도','경상북도','강원도','전라남도'], col_years]
# df_3.head()
df_4 = df_4.T
df_4.plot(kind='bar',width=0.7,color=['orange','green','skyblue','blue'], figsize = (10,5))

plt.title('Seoul to')
plt.xlabel('Period')
plt.ylabel('Immigration No.')

plt.show()


# In[225]:


# 세로 막대그래프로 표현
col_years = list(map(str, range(2001,2018)))

df_4 = df_seoul.loc[['충청남도','경상북도','강원도','전라남도'], col_years]
# df_3.head()
df_4 = df_4.T
df_4.plot(kind='barh',width=0.7,color=['orange','green','skyblue','blue'], figsize = (10,5))

plt.title('Seoul to')
plt.xlabel('Period')
plt.ylabel('Immigration No.')

plt.show()


# In[214]:


# 합계를 만들어서 세로 막대그래프로 표현
col_years = list(map(str, range(2001,2018)))

df_5 = df_seoul.loc[['충청남도','경상북도','강원도','전라남도'], col_years]
df_5['합계'] = df_5.sum(axis=1)
df_total = df_5[['합계']].sort_values(by='합계')
print(df_5)
df_total.plot(kind='barh',figsize=(8,4))
plt.legend(loc='best')
plt.title('Seoul to')
plt.xlabel('Period')
plt.ylabel('Immigration No.')

plt.show()


# In[226]:


# Excel 데이터를 데이터프레임 변환 
df = pd.read_excel('./test_data/남북한발전전력량.xlsx', engine= 'openpyxl', convert_float=True)
df = df.loc[5:9]
df.drop('전력량 (억㎾h)', axis='columns', inplace=True)
df.set_index('발전 전력별', inplace=True)
df = df.T 

# 증감율(변동률) 계산
df = df.rename(columns={'합계':'총발전량'}) # '합계' 컬럼명을 총발전량으로 변경
df['총발전량 - 1년'] = df['총발전량'].shift(1) # shift(1)은 전년도 총발전량을 의미함.
df['증감율'] = ((df['총발전량'] / df['총발전량 - 1년']) - 1) * 100      

# 2축 그래프 그리기
ax1 = df[['수력','화력']].plot(kind='bar', figsize=(15, 7), width=0.7, stacked=True)  
ax2 = ax1.twinx()
ax2.plot(df.index, df.증감율, ls='--', marker='o', markersize=20, 
         color='green', label='전년대비 증감율(%)')  

ax1.set_ylim(0, 500)
ax2.set_ylim(-50, 50)

ax1.set_xlabel('연도', size=20)
ax1.set_ylabel('발전량(억 KWh)')
ax2.set_ylabel('전년 대비 증감율(%)')

plt.title('북한 전력 발전량 (1990 ~ 2016)', size=30)
ax1.legend(loc='upper left')

plt.show()


# In[ ]:


# plot 생성 : 사이즈, 몇개의 그래프를 그릴건지?
# title, legend, x축, y축, 그래프타입, xlabel, ylabel
# annotate, ax1.twinx() : x축을 공유해서 그래프를 그림


# In[220]:


# read_csv() 함수로 df 생성
df = pd.read_csv('./test_data/auto-mpg.csv', header=None)

# 열 이름을 지정
df.columns = ['mpg','cylinders','displacement','horsepower','weight',
              'acceleration','model year','origin','name']

# 연비(mpg) 열에 대한 히스토그램 그리기
df['mpg'].plot(kind='hist', bins=10, color='coral', figsize=(10, 5))

# 그래프 꾸미기
plt.title('Histogram')
plt.xlabel('mpg')
# cylinders 개수의 상대적 비율을 계산하여 시리즈 생성
cylinders_size = df.cylinders / df.cylinders.max() * 300

# 3개의 변수로 산점도 그리기 
df.plot(kind='scatter', x='weight', y='mpg', c='coral', figsize=(10, 5),
        s=cylinders_size, alpha=0.3)
plt.title('Scatter Plot: mpg-weight-cylinders')

# cylinders 개수의 상대적 비율을 계산하여 시리즈 생성
cylinders_size = df.cylinders / df.cylinders.max() * 300

# 3개의 변수로 산점도 그리기 
df.plot(kind='scatter', x='weight', y='mpg', marker='+', figsize=(10, 5),
        cmap='viridis', c=cylinders_size, s=50, alpha=0.3)
plt.title('Scatter Plot: mpg-weight-cylinders')

plt.savefig("./scatter.png")   # 그래프 파일로 저장
plt.savefig("./scatter_transparent.png", transparent=True)   

plt.show()


# In[219]:


# 데이터 개수 카운트를 위해 값 1을 가진 열을 추가
df['count'] = 1
df_origin = df.groupby('origin').sum()   # origin 열을 기준으로 그룹화, 합계 연산
print(df_origin.head())                  # 그룹 연산 결과 출력

# 제조국가(origin) 값을 실제 지역명으로 변경
df_origin.index = ['USA', 'EU', 'JAPAN']

# 제조국가(origin) 열에 대한 파이 차트 그리기 – count 열 데이터 사용
df_origin['count'].plot(kind='pie', 
                     figsize=(7, 5),
                     autopct='%1.1f%%',   # 퍼센트 % 표시
                     startangle=10,       # 파이 조각을 나누는 시작점(각도 표시)
                     colors=['chocolate', 'bisque', 'cadetblue']    # 색상 리스트
                     )

plt.title('Model Origin', size=20)
plt.axis('equal')    # 파이 차트의 비율을 같게 (원에 가깝게) 조정
plt.legend(labels=df_origin.index, loc='upper right')   # 범례 표시
plt.show()


# In[222]:


# 그래프는 데이터의 분포도를 시각화해서 볼 수 있도록
 pip install folium


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
