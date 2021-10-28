#!/usr/bin/env python
# coding: utf-8

# In[17]:


print("hello python")
a = 10
b = 20
a
a + b


# In[18]:


print("파이썬")


# - 노트북 파일 확장자 ~.ipynb => jupyter notebook
# - 일반파이썬 파일 확장자 ~.py => python 파일명
# 
# 줄바꿈  
# 줄바꿈
# 

# In[19]:


# 키워드 출력
import keyword #이것은 주석

print(keyword.kwlist, 'test')


# In[20]:


# print 연습
# 하나만 출력합니다.
print("# 하나만 출력합니다.")
print("hello python Programmimg...!")
print()

# 여러 개를 출력합니다.
print("# 여러 개를 출력합니다.")
print("Hello", 'Good', 'Morning!!')
print()

# 아무것도 입력하지 않으면 단순하게 줄바뀜
print("# 아무것도 입력하지 않으면 단순하게 줄바뀜")
print('--- Line ---')
print()
print()
print('--- Line ---')


# In[21]:


# \n :new line ,\t : tab
print()
print("안녕하세요\n안녕하세요")
print()
print("name\tage\tlocation")
print("ken\t19\tUSA")
print()
#\""
print("\"안녕하세요\"")


# In[22]:


# 여러 라인의 문자열 만들기
print("동해물과 백두산이\n마르고 닳도록\n하느님이")
print()

print("""동해물과 백두산이
마르고닳도록
하느님이""")
print()
print('''동해물과 백두산이
마르고닳도록
하느님이''')
print()
a = '''동해물과 백두산이
마르고닳도록
하느님이'''

print(a)


# In[23]:


# 문자열 연산자 : + , *
# 문자열 + 문자열 = 문자열 이어 붙이기
# 문자열 * 숫자 = 문자열을 숫자만큼 반복

print("hello " + "python!")
print("hello " * 3)


# In[24]:


# 문자열 index 및 slice 
# -> str[0] : 0번지 값
# -> str[1:4] -> start : end : start 포함, end는 포함하지 않음.

string_1 = "안녕하세요"

print(string_1)
print(string_1[0],string_1[4]) # 안요
print(string_1[-1],string_1[-3]) #요하
print(string_1[1:3]) # 녕하
print(string_1[:3])
print(string_1[3:])
print(len(string_1))


# In[25]:


# str = 임의의 문자를 넣으세요,
# str의 길이를 구해서 처음 문자에 문자를 음수 인덱스로 구해서 출력하세요.

string_0 = "가나다라마바사아자차카파"
print(string_0[0]," : ",str[-len(string_0)])


# In[26]:


# 숫자 : 정수, 실수
# 연산자 : + - / * // % **
print("5+5=",5+5)
print("4-3=",4-3)
print("8*3=",8*3)
print("8//3=",8//3)
print("8%3=",8%3)
print("8**3=",8**3)
print("8/3=",8/3)


# In[33]:


# 변수 : 값을 저장하는 공간, 메모리에 방을 만듬.

pi = 3.14
print(int(pi*2))
print(pi-2)
print(pi*2)

print("string" + "10")
str(10) + "abcd"


# In[28]:


# 복합대입 연산자
a = 0
a = a + 1
a = a + 1
a = a + 1
print(a)

b = 0
b += 1
b += 1
b += 1
print(b)


# In[29]:


str = "God" ; str+= " test" ; str *= 2
print("str : ",str)


# In[30]:


# 문자열을 입력받아 문자열의 길이를 출력, 처음 글자와 마지막 글자를 출력핫요,

str = input()

print("문자의 길이는 ",len(str),"입니다.")
print("문자의 처음글자는 ",str[0],"입니다.")
print("문자의 마지막 글자는 ",str[len(str)-1],"입니다.")


# In[34]:


# 숫자 2개를 입력 받아 정수형으로 바꾸고
# 두 수의 사칙연산을 출력하세요.
# 문자열을 입력받아 처음 입력받은 숫자만큼 반복해서 출력하세요.

num_1 = input("첫번째 숫자 입력 : ")
num_2 = input("두번째 숫자 입력 : ")

num = int(num_1) * int(num_2) + int(num_2)

print("사칙연산 결과는 ",num,"입니다.")

string_0 = input("문자열을 입력 : ")
print("반복출력")
print(string_0 * num)


# In[35]:


# inch 단위의 자료를 입력받아 cm 단위를 구하는 코드

str_input = input("숫자입력>")
num_input = int(str_input)

print()
print(num_input, "inch")
print(num_input * 2.54, "cm")


# In[42]:


# 원의 반지름을 입력 받아 원의 둘레와 넓이를 구하는 코드

str_length = input("반지름 길이입력>")
num_length = float(str_length)

print("원의둘레 :",num_length * 2 * 3.14)
print("원의 넓이 :",3.14 * num_length ** 2)


# In[47]:


# 프로그램을 실행했을 때, 문자열 2개를 입력을 받고 순서를 바꿔서 출력, 안바꾸고 각각 출력하는 코드

a = input("문자열 : ")
b = input("문자열 : ")

print(a,b)
print(b,a)

print("변수를 스왑하여 출력")

print(a,b)

c = a
a = b
b = c

print(a,b)


# In[ ]:




