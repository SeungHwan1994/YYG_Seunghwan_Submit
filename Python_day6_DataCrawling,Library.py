#!/usr/bin/env python
# coding: utf-8

# ### 학습목표
# 1. beautifulsoup 모듈 사용하기

# In[2]:


from bs4 import BeautifulSoup


# #### html 문자열 파싱
#  - 문자열로 정의된 html 데이터 파싱하기

# In[3]:


html = '''
<html>
  <head>
    <title>BeautifulSoup test</title>
  </head>
  <body>
    <div id='upper' class='test' custom='good'>
      <h3 title='Good Content Title'>Contents Title</h3>
      <p>Test contents</p>
    </div>
    <div id='lower' class='test' custom='nice'>
      <p>Test Test Test 1</p>
      <p>Test Test Test 2</p>
      <p>Test Test Test 3</p>
    </div>
  </body>
</html>'''


# #### find 함수
#  - 특정 html tag를 검색
#  - 검색 조건을 명시하여 찾고자하는 tag를 검색

# In[4]:


soup = BeautifulSoup(html)


# In[5]:


soup.find('h3')


# In[6]:


soup.find_all('p')


# In[7]:


soup.find('div', class_ ='test')


# In[8]:


attrs = {'id': 'upper', 'class': 'test'}
soup.find('div', attrs=attrs)


# #### find_all 함수
#  - find가 조건에 만족하는 하나의 tag만 검색한다면, find_all은 조건에 맞는 모든 tag를 리스트로 반환

# In[9]:


soup.find_all('div', class_='test')


# #### get_text 함수
#  - tag안의 value를 추출
#  - 부모tag의 경우, 모든 자식 tag의 value를 추출

# In[10]:


tag = soup.find('h3')
print(tag)
tag.get_text()


# In[11]:


tag = soup.find('p')
print(tag)
tag.get_text()


# In[15]:


tag = soup.find('div', id='upper') # 부모밑에 자식 태그가 존재
print(tag)
tag.get_text()
#tag.get_text().strip()


# #### attribute 값 추출하기
#  - 경우에 따라 추출하고자 하는 값이 attribute에도 존재함
#  - 이 경우에는 검색한 tag에 attribute 이름을 [ ]연산을 통해 추출가능
#  - 예) div.find('h3')['title']

# In[13]:


tag = soup.find('h3')
print(tag)
tag['title']
#tag.get_text()


# In[17]:


from urllib import request
from bs4 import BeautifulSoup

#영화 랭킹 검색하기, 순위대로 평점 출력하기
url = "https://movie.naver.com/movie/sdb/rank/rmovie.naver?sel=cur&date=20211103"
target = request.urlopen(url)

soup = BeautifulSoup(target) # <html> tag로 변경

point_all = soup.find("td", class_ = "point") 
print(point_all.get_text())

# find의 목적은 원하는 태그를 찾는 것이다
# select는 CSS selector로 tag 객체를 찾아 반환한다.

# get_text() : 한 태그당 텍스트 추출함.
# <div calss_='tit5'> : 영화제목
# >td calss_='point'> : 평점
# soup.find('tr'), soup.findAll(찾고자하는 tag 입력, class='클래스명')

# for title in soup.select("title")


# In[22]:


movie_title = []
movie_point = []

# 영화제목과 평점 리스트 만들기
for line in soup.find_all('tr'):
    title = line.find("div", class_="tit5") # 클래스는 언더라인 해야함
    if title: # 영화제목 리스트에 추가
        movie_title.append(title.get_text().strip("\n")) # \n 제거
    point = line.find('td', class_='point')
    if point: # 값이 없으면 처리문 X 값이 있으면 처리문 진행
        movie_point.append(point.get_text())
        
for i, (title, point) in enumerate(zip(movie_title,movie_point)):
    print("{} : {} - {}".format(i + 1, title, point))


# In[23]:


import os
import sys
import urllib.request
client_id = "krtqG7yTrcUyvJF45IVn" # 개발자센터에서 발급받은 Client ID 값
client_secret = "UpnnuDqOBO" # 개발자센터에서 발급받은 Client Secret 값
encText = urllib.parse.quote("반갑습니다")
data = "source=ko&target=en&text=" + encText
url = "https://openapi.naver.com/v1/papago/n2mt"
request = urllib.request.Request(url)
request.add_header("X-Naver-Client-Id",client_id)
request.add_header("X-Naver-Client-Secret",client_secret)
response = urllib.request.urlopen(request, data=data.encode("utf-8"))
rescode = response.getcode()
if(rescode==200):
    response_body = response.read()
    print(response_body.decode('utf-8'))
else:
    print("Error Code:" + rescode)


# In[24]:


curl "https://openapi.naver.com/v1/papago/n2mt"     -H "Content-Type: application/x-www-form-urlencoded; charset=UTF-8"     -H "X-Naver-Client-Id: {애플리케이션 등록 시 발급받은 클라이언트 아이디 값}"     -H "X-Naver-Client-Secret: {애플리케이션 등록 시 발급받은 클라이언트 시크릿 값}"     -d "source=ko&target=en&text=만나서 반갑습니다." -v


# In[ ]:


from urllib import request


# In[26]:


text = '''Yesterday
All my troubles seemed so far away
Now it looks as though they're here to stay
Oh, I believe in yesterday
Suddenly
I'm not half the man I used to be
There's a shadow hanging over me
Oh, yesterday came suddenly
Why she had to go, I don't know
She wouldn't say
I said something wrong
Now I long for yesterday
Yesterday
Love was such an easy game to play
Now I need a place to hide away
Oh, I believe in yesterday
Why she'''


# In[ ]:


import os
import sys
import urllib.request
client_id = "krtqG7yTrcUyvJF45IVn" # 개발자센터에서 발급받은 Client ID 값
client_secret = "UpnnuDqOBO" # 개발자센터에서 발급받은 Client Secret 값
encText = urllib.parse.quote(text)
data = "source=en&target=ko&text=" + encText
url = "https://openapi.naver.com/v1/papago/n2mt"
request = urllib.request.Request(url)
request.add_header("X-Naver-Client-Id",client_id)
request.add_header("X-Naver-Client-Secret",client_secret)
response = urllib.request.urlopen(request, data=data.encode("utf-8"))
rescode = response.getcode()
if(rescode==200):
    response_body = response.read()
    print(response_body.decode('utf-8'))
else:
    print("Error Code:" + rescode)


# In[92]:


print(type(response))
print(type(response_body))
print(type(response_body.decode('utf-8')))
print(type(json.loads(response_body.decode('utf-8'))))


print(json.loads(response_body.decode('utf-8'))
      ["message"]["result"]["translatedText"])
# response_body는 byte

print("---------------------raw string------------------------")
print(response_body.decode('utf-8'))
# json loads -> 문자열을 dictionary로 변환한다. / load 는 파일을 읽을때 사용함.


# In[61]:


import re
m = re.search('"translatedText":(.+?),engineType"',
              response_body.decode('utf-8'))
if m:
    found = m.group(0)

print(found)


# In[95]:


"""
웹 검색하는 방법
1, 브라우저에서 자료를 가져오는 방법
2, api를 활용해서 가져오는 방법
    해당  사이트에서 개발자 가이드를 제공
    ID와 비밀번호를 제공한다. (혹은 인증키를 제공)
    가이드에 따라 원하는 정보를 요청
    일반적으로 json 혹은 XML 형식으로 데이터를 제공 : python에서는 json으로 받음
    가지고온 자료의 타입을 확인해서 python 데이터타입으로 변경
    json -> dictionary로 변경 가능

- 데이터 크롤링 -
1. urlopen
2. 데이터 파싱, 가공 -> 데이터 생성
3. 데이터를 가지고 분석

"""


# In[12]:


# test package의 모듈을 추가
# 패키지의 모듈을 추가 : import 패키지명,모듈명
# 라이브러리 = 폴더 / 모듈 = 파이썬파일(.py)

import test_package.module_a as a
import test_package.module_b as b

print(a.variable_a)
print(b.variable_b)


# In[8]:


from test_package import * # 해당 패키지 디렉토리 안에 __init__.py 가 존재해야함.

# __init__.py
# __all__ = ['module_a', 'module_b']

print(module_a.variable_a)
print(module_b.variable_b)


# >패키지가 저장되는 경로 : C:\Users\lshwa\anaconda3\Lib\site-packages
# >패키지는 디렉토리로 생성해서 그 밑에 모듈관리
# >from 패키지명 import * -> 함수명과 함수명 중복 우려로 사용되지않음.

# In[ ]:





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
