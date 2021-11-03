#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Python의 데이터 타입 : 문자, 숫자, 불리언

자료를 모아놓아 사용하는 데이터 : list[]

    list를 접근하는 방법
    -> index 첨자로 접근하는 방법 : list[1:3], list[ : ]
    -> list.append() / list.insert() / list.extend() / del list[1]
    
키와 값을 저장하는 데이터 : dictionary{}
    
    dic = {'a' : 10, 'b' : 20}
    -> dic[key] / dic['c'] = 'abcd'
    del dic[key], dic.clear()
    
file (from 하드디스크)로 접근하는 방법
    file = open(파일명, 모드) -> 파일명은 경로포함, 모드 : w, r, a, w+, r+, a+. wb...
    text = 'abcd' -> text 내요을 파일에 저장 : file.write(text)
    file.close()
    
반복분
    for 식별자 in 반복되는 값:
    [값 for 식별자 in 반복되는 값 if 표현식]
    while 표현식:
"""


# In[2]:


print([a for a in range(10) if a < 5])


# In[2]:


# 함수 : 정의부분과 실행하는 부분
# 정의 : 함수 작성하면서 기능을 구현

# def 함수명():
#     실행문 ....
    
# 실행 : 함수명()
    
def num_total(start_nom, end_nom):
    total = 0
    for num in range(start_nom,end_nom + 1):
        total += num
    return total

print(num_total(1,100))


# In[33]:


# 이름을 입력받아 list_name에 저장 : 입력의 끝은 'q' ->input_name_func()으로 구현
def input_name_func(list_name):
    while True:  
        input_data = input("name >")
        if input_data == 'q':
            break
        list_name.append(input_data)
# 입력된 이름의 개수만큼 성적을 입력 받아 list_score에 저장 ->input_score_func()으로구현
def input_score_func(list_score,n):
    while True:
        if len(list_score) >= n:
            break
        input_data = input("score >")
        list_score.append(int(input_data))
# 입력된 이름을 키로, 성적을 값으로 저장하는 함수 작성 -> dict_name_score 식별자,
# make_dict_func()으로 구현

def make_dict_func():
    dict_name_score = {}
    list_name = []
    list_score = []
    
    input_name_func(list_name)
    input_score_func(list_score,len(list_name))

    for i in range(len(list_name)):
        key = list_name[i]
        value = list_score[i]
        dict_name_score[key] = value        
    return dict_name_score

# 저장된 dict_name_score의 자료를 파일로 저장 -> 구분자는 ',' -> 홍길동,90
# 파일명은 file.txt로 저장 "w" 모드로 open / file close 
def file_write(dict_name_score):
    file  = open('./file.txt','w')
    for key,value in dict_name_score.items():
        file.write("{},{}\n".format(key,value))
    file.close()
    
# 저장된 파일을 'r' 모드로 open해서 파일의 내용을 dict로 저장
def file_to_dict():
    file_to_dict = {}
    file = open("./file.txt",'r')
    file.seek(0)
    for i in file:
        list_i = i.split(",")
        file_to_dict[list_i[0]] = int(list_i[1])
    file.close()
    return file_to_dict

#함수의 실행
dict_name_score = make_dict_func()
print("dict_name_score :",dict_name_score)

file_write(dict_name_score)
new_dict = file_to_dict()
print("new_dict :",new_dict)


# In[36]:


# 검색하고자 하는 이름을 입력
# 파일의 내용에서 같은 이름을 검색해서 점수를 출력
# 이름이 중복된 경우 처음 점수만 출력

s_name = input("검색할 이름 >")
dict_name_score = file_to_dict()
s_score = 0
for name in dict_name_score:
    if name == s_name:
        print("{}의 점수는 : {}".format(name,dict_name_score[name]))
        s_score = dict_name_score[name]
        break
    
if s_score == 0:
    print("존재하지않는 이름입니다.")


# In[3]:


# 모듈 : 소스와 분리되어 저장된 여러 변수와 함수의 집합체

import math # 전체 모듈 불러오기
import math as mt # 모듈의 식별자를 변경하기

print(mt.ceil(34.5))
print(math.floor(34.5))


# In[2]:


# from 모듈이름 import 변수 또는 함수  : 모듈에서 특정 변수 및 함수만 가져오기

from math import sin, cos, ceil

print(sin(10))

tan(10)


# In[53]:


import random

print("int(random.random() * 100) :",int(random.random() * 100))

# uniform(min, max) : 지정한 범위 사이의 float을 리턴함.
print("random.uniform(1,2) :",random.uniform(1,2))

# randrange(max / min, max) : 지정한 범위의 int를 리턴함. (기본값 min = 0)
print("random.randrange(5,7) :",random.randrange(5,7))

# choice(list) : 리스트의 요소를 랜덤하게 선택
list_45 = [i for i in range(1,45 + 1)]
print(list_45)
print("random.choice(list_45) :",random.choice(list_45))

# shuffle(list) : 리스트의 요소들을 랜덤하게 섞음. -< 리턴값은 None
random.shuffle(list_45)
print(list_45)

# sample(list, k = int) : 리스트 요소중에 k개 뽑음
print("random.sample(list_45,k=3) :",random.sample(list_45,k=3))


# In[72]:


list_45 = [i for i in range(1,45 + 1)]

target_list = random.sample(list_45,k=6)
target_list.sort()
try_number = 0
print("목표 당첨 숫자는 :",target_list)

while True:
    try_list = random.sample(list_45,k=6)
    try_list.sort()
    try_number += 1
    if try_list == target_list:
        break


print("복권의 시도 횟수는 :",try_number)

#8098808
#4496718
# 목표 당첨 숫자는 : [8, 19, 22, 30, 41, 44]
# 복권의 시도 횟수는 : 4445814


# In[55]:


# import os 운영체제와 관련된 기능을 사진 모듈
import os

print("현재 운영체제 :", os.name)
print("현재 작업폴더 :", os.getcwd())
print("현재 작업폴더의 디텍토리 :", os.listdir())

# os.mkdir("dir name") -> 디렉토리 생성
# os.remove("file name") -> 파일 삭제
# os.system(명령어) -> dos 에서 해당 명령어 실행

import time
time.sleep(5) # 5초 동안 연산은 중지함.
 


# In[71]:


# urllib 모듈 : URL 다루는 라이브러리
from urllib import request

web_page = request.urlopen("https://www.google.com/")
output = web_page.read()

print(output) # beautifulsoup 사용하기전


# In[83]:


# !pip install beautifulsoup4 -> 외부모듈 설치

from bs4 import BeautifulSoup

target = request.urlopen("http://www.ycampus.co.kr/")

soup = BeautifulSoup(target)
# print(soup)

for title in soup.select("title"): # title이 포함된 line만 출력하기
    print(title)


# In[84]:


from urllib import request
from bs4 import BeautifulSoup

# urlopen() 함수로 기상청의 전국 날씨를 읽습니다.
target = request.urlopen(
    "http://www.kma.go.kr/weather/forecast/mid-term-rss3.jsp?stnId=108")

# BeautifulSoup을 사용해 웹 페이지를 분석합니다.
soup = BeautifulSoup(target, "html.parser") # xml -> html으로 변환

# location 태그를 찾습니다.
for location in soup.select("location"):
    # 내부의 city, wf, tmn, tmx 태그를 찾아 출력합니다.
    print("도시:", location.select_one("city").string)
    print("날씨:", location.select_one("wf").string)
    print("최저기온:", location.select_one("tmn").string)
    print("최고기온:", location.select_one("tmx").string)
    print()


# In[87]:


# !pip install flask

from flask import Flask

app = Flask(__name__)

@app.route("/")
def hello():
    return "<h1>Hello World! </h1"


# In[ ]:


# urlopen -> urllib -> request.open('url명')
# BeautifulSoup -> url 내용을 <html> tag로 변경 해주는 함수

from urllib import request
from bs4 import BeautifulSoup

#1. url open
target = request.open("url경로")
soup = BeautifulSoup(target, "html.parser") # <html tag> 변환

# 웹브라우저를 활용한 프로그램 Django 나 Flask를 사용해야함.

from flask import Flask


# In[ ]:





















