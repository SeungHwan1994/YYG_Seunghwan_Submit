#!/usr/bin/env python
# coding: utf-8

# In[3]:


# 키보드로부터 입력을 받아 해당 문자열을 출력하고 길이를 구하세요.

string_0 = input("문자열 입력>")

print("입력받은 문자 :",string_0)
print("입력받은 문자의 길이는 :",len(string_0),"입니다.")


# In[10]:


# format : {}에 값을 대입함,

print("{}".format(10))   # 10

print("{} {} {}".format(10,"test",20)) # 10 test 20

print("입력받은 문자 : {} \n문자의 길이는 {}".format(string_0,len(string_0)))


# In[12]:


# 정수 자릿수 지정하기 {:숫자d}

num_0 = 52
print("{:d}".format(num_0))

print("{:5d}".format(num_0))

print("{:05d}".format(num_0))

print("{:05d}".format(num_0*-1))

print("{:5d}".format(num_0*-1))


# In[18]:


# 부호 붙이기 {:+d} = 양수든 음수든 부호를 붙임.
# {: d} = 양수를 부호를 붙이고, 음수를 부호를 안붙임.

print("{:+d}".format(num_0))
print("{:+d}".format(num_0*-1)) 
print("{: d}".format(num_0))
print("{: d}".format(num_0*-1)) 


# In[20]:


# 조합하기 {:=d} 맨앞자리에 부호를 붙임.

print("{:+5d}".format(num_0))
print("{:+5d}".format(num_0*-1))
print("{:=+5d}".format(num_0))
print("{:=+5d}".format(num_0*-1))
print("{:=+05d}".format(num_0))
print("{:=+05d}".format(num_0*-1))


# In[48]:


# 부동소수 점 출력 {:f}
print("부동소수점 출력")
print()
num_f = 52.273
print("{:f}".format(num_f))
print("{:15f}".format(num_f)) # 전체 표현하는 자리수가 15개, 소수점 전체 표시
print("{:+15f}".format(num_f)) 
print("{:+015f}".format(num_f))

#소숫점 미만 표현 자리수 지정 {:15.3f} 전체 표현 자릿수는 15, 소수점 미만 3자리
print("소수점 미만 표현 자리수 지정")
print()
print("{:15.3f}".format(num_f)) # 전체 표현 자릿수는 15, 소수점 미만 3자리

# 의미없는 소수점 제거하기 {:g}
print("의미없는 소수점 제거하기")
print()
print("{:g}".format(10.00000))


# In[41]:


# upper 함수, lower 함수
# 사용하는 문법 : [문자열].[함수()]


string_a = '   Hello World !!   '

print(string_a.upper()) # 모든 문자열을 대문자로 변환
print(string_a.lower()) # 모든 문자열을 소문자로 변환
print(string_a.strip()) # 양쪽 공백 제거
print(string_a.lstrip()) # 왼쪽 공백 제거
print(string_a.rstrip()) # 오른쪽 공백 제거

# 대소문자 변경 후 공백 제거하기

print(string_a.lower().strip()) # 소문자 변경 후 공백제거
print(string_a.upper().strip()) # 대문자로 변경 후 공백제거


# In[47]:


# 문자열 구성 파악하기 is함수명() -> True or False

print("Translate".isnumeric()) # 숫자인가?
print("Translate".isalnum())
print("10".isdigit())
print("import".isidentifier()) #  식별자로 사용할 수 있는지?
print("import A as".isidentifier()) #  식별자로 사용할 수 있는지?


# In[54]:


# find() 와 rfind() : 찾고자하는 문자열의 위치를 반환함.

input_str = "hellohello python" # 처음 나오는 hello를 추출해서 출력

start_p = input_str.find("hello") # 스타트 위치를 찾는다
end_p = input_str.rfind('hello') # 끝나는 위치를 찾는다

print("시작위치 : {}, 마지막 위치 : {}, {}"
      .format(start_p,end_p,input_str[start_p:end_p]))

# 마지막 위치의 hello부터 뒤에 있는 문자 모두 출력

print(input_str[end_p:])


# In[58]:


# 키보드에서 공백 포함해서 문자열을 입력 받아
# 처음 공백 나오는 위치까지의 문자만 출력 abcd test -> abcd만 출력

string_ip = input("문자열을 입력>")
print( string_ip [ : string_ip.find(" ") ] )

# 함수명() -> len(), print(), type()
# 오브젝트.함수명() -> 클래스 ( 맴버변수, 매서드(함수) ) -> str, upper,...


# In[71]:


# 문자열 자르기 문자열.split()

a_str = "10 20 30 40 test abcd" # 공백으로 분리
a = a_str.split()

print(a) # 리스트로 출력함.

b_str = "10,20,30,40,test,abcd" # , 으로 분리
b = b_str.split(",")

print(b) # 리스트로 출력함

# 키보드로 부터 데이터를 입력받아 공백으로 데이터를 분리하고
# 2번째 자료를 출력하세요
# abcd test c d 10 30 -> test를 출력

string_ip2 = input("공백으로 나눈 문자열 입력> ")
list_split = string_ip2.split()

print("입력한 리스트는",list_split,
      "\n입력한 문자열에서 두번째 자료의 값은 : {}".format(list_split[1]))


# In[81]:


# and, or, not , == , !=, <, >, <=, >=

a = 10 ; b = 20

print(a == b) # F
print(a != b) # T
print(a < b) # T
print(a > b) # F
print(a >= b) # F
print(a <= b) # T

print(a == 10 and b == 20) # T
print(a != 10 and b == 20) # F
print(a == 10 and b != 20) # F
print(a != 10 and b != 20) # F

print(a == 10 or b == 20) # T
print(a != 10 or b == 20) # T
print(a == 10 or b != 20) # T
print(a != 10 or b != 20) # F

print(not (a != 10 or b != 20))  # T

print("in 연산자 결과")
print('test' in "10.20.30.40.test.abcd")


# In[92]:


# if 조건식 :
#     들여쓰기

if True:
    print("if 문장 실행")
    
if False:
    print("if False 실행") # 실행되지않음.
    


# In[93]:


# 정수를 입력받아 10보다 작으면 입력된 값 출력
# 1. 정수입력 받음
# 2. 숫자인지 확인 isnumeric()
# 3. 숫자이면 문자를 숫자로 변경
# 4. 10보다 작으면 출력

input_int = input("정수입력> ")

if input_int.isdecimal() and float(input_int) < 10:
    print("input_int : {:+5d}".format(float(input_int)))
print("end")

# Q : 왜 음수를 넣으면 안되는가?


# In[99]:


# 숫자를 입력받아 실수로 변환한 후
# 0보다 크면 '양수, 0보다 작으면 ' 음수, 0이면 'ZERO'를 출력하세요.

input_if = float(input("숫자입력> "))

if input_if > 0 :
    print("양수")
    
if input_if < 0 :
    print("음수")
    
if input_if == 0 :
    print("ZERO")
    
# 숫자인 경우 0이면 False / 문자열인 경우 none 이면 False

a = ''
if not a:
    print("a는 False")


# In[6]:


input_num = input("정수입력> ")
last_char = input_num[-1]
last_num = int(last_char)

#마지막 글자를 활욜한 홀짝 구분법
print("in연산자 활용")
if last_char in "02468":
    print("짝수입니다.")
    
if last_char in "13579":
    print("홀수입니다\n")
    
#나머지 값을 활용한 홀짝 구분법
print("나머지활용")
if int(input_num) % 2 == 0:
    print("짝수입니다.")
if int(input_num) % 2 == 1:
    print("홀수입니다.")


# In[21]:


# 1. 공백으로 구분한 두 개의 숫자를 입력받아 작은 수부터 출력
# 2. 작은 수와 큰 수 각각 3의 배수인지 확인, 3의 배수이면 "3의 배수" 출력
#    아니면 "3의 배수가 아닙니다." 출력"
# 34 57 입력하면 -> 34 57 출력하고, 34는 3의 배수가 아닙니다. 57은 3의 배수입니다.

input_char = input("두 개의 정수 입력> ")
input_1 = int(input_char.split(" ")[0])
input_2 = int(input_char.split(" ")[1])

if input_1 <= input_2:
    num_1 = input_1
    num_2 = input_2
    
    print("입력된 정수는 {}와 {}입니다.".format(num_1,num_2))
    
    if num_1 % 3 == 0:
        print(num_1,"은 3의 배수입니다.")
    if num_1 % 3 != 0:
        print(num_1,"은 3의 배수가 아닙니다.")
    
    if num_2 % 3 == 0:
        print(num_2,"은 3의 배수입니다.")
    if num_2 % 3 != 0:
        print(num_2,"은 3의 배수가 아닙니다.")
    
if input_1 > input_2:
    num_1 = input_2
    num_2 = input_1

    print("입력된 정수는 {}와 {}입니다.".format(num_1,num_2))
    
    if num_1 % 3 == 0:
        print(num_1,"은 3의 배수입니다.")
    if num_1 % 3 != 0:
        print(num_1,"은 3의 배수가 아닙니다.")
    
    if num_2 % 3 == 0:
        print(num_2,"은 3의 배수입니다.")
    if num_2 % 3 != 0:
        print(num_2,"은 3의 배수가 아닙니다.")


# In[3]:


# 정수를 입력받아 90이상이면 'A', 80이상이면 'B', 70이상이면 'C', 그외 'D'

input_number = int(input("점수 입력> "))

if input_number >= 90:
    print('A')
elif input_number >= 80:
    print('B')
elif input_number >= 70:
    print('C')
else:
    print('D')
    


# In[12]:


# 낳짜 함수에서 월을 추출해서 현재 월이 봄인지, 가을인지 출력
# 1~3월은 봄 4~6월은 여름 7~9월은 가을 10~12월은 겨울이라고 출력
# if ~ elif ~ else 사용한다.

import datetime

now = datetime.datetime.now() # 모듈명(datetime) - 클래스명(datetime) - 메서드(.now)

if 1 <= now.month < 4:
    season = "봄"
elif 4 <= now.month < 7:
    season = "여름"
elif 7 <= now.month < 10:
    season = "가을"
else:
    season = "겨울"

print("현재는 {}월이고, 계절은 {}입니다.".format(now.month,season))

# if , elif , else 등 처리문장은 반드시 처리가 있어야 에러가 발생하지않음.
# pass를 사용하면 처리가 없어도 에러가 발생하지 않고 넘어간다.

if True:
    pass


# In[5]:


## 숫자 부호 숫자를 입력받아 계산하는 프로그램 작성
# 10 + 20 = 30 / 10 - 20 = -10, 부호는 +, -, *, / 를 입력하세요.
# 10 + 20 = 30 이렇게 출력 할 수 있도록.

input_raw = input("계산할 식 입력 (형식 : 10 + 20)> ")
num_1 = int(input_raw.split()[0])
math_0 = input_raw.split()[1]
num_2 = int(input_raw.split()[2])
output = ""

if math_0 == "+":
    output = num_1 + num_2
elif math_0 == "-":
    output = num_1 - num_2
elif math_0 == "*":
    output = num_1 * num_2
elif math_0 == "/" and num_2 != 0:
    output = num_1 / num_2

if math_0 == "/" and num_2 == 0 :
    print("0은 나눌 수 없습니다.")
elif output == "":
    print("사칙연산만 가능합니다")
else:
    print("{} {} {} = {}".format(num_1,math_0,num_2,output))


# In[20]:


# in 연산자를 이용해서 더 깔끔하게 코딩해보아라.

input_raw = input("계산할 식 입력 (형식 : 10 + 20)> ")
num_1 = int(input_raw.split()[0])
math_0 = input_raw.split()[1]
num_2 = int(input_raw.split()[2])

if math_0 in "+-/*":
    if num_2 != 0:
        print(num_1 math_0 num_2)


# In[14]:


# list = [ , , , , , ] -> 여러개의 데이터 집합
# +, *, len() : 리스트의 개수
# .append() : 끝에 요소 추가 / .insert(위치,요소) : 특정위치에 요소 추가

a_list = [1, 2, 3]
b_list = ['a', 'b', 'c']

print("a_list + b_list = ",a_list + b_list)
print()

print("a_list * 3 = ",a_list * 3)
print()

c_list = a_list + b_list

print("a_list 개수 : {}, b_list 개수 : {}, c_list 개수 : {}"
     .format(len(a_list),len(b_list),len(c_list)))
print()


a_list.append(b_list)
print("a_list에 b_list append:",a_list)
print("a_list의 4번째 요소 리스트의 첫번째요소 :",a_list[3][0])

a_list.insert(1,b_list)
print(a_list)


# In[17]:


# if 조건식 : 처리문 else : 처리문

input_num = input("정수입력> ")
last_char = input_num[-1]
last_num = int(last_char)

#나머지 값을 활용한 홀짝 구분법
print("나머지활용")
if int(input_num) % 2 == 0:
    print("짝수입니다.")
else:
    print("홀수입니다.")


# In[41]:


# a_list = [1, 2, 3] , b_list = ["test", "abcd"]
a_list = [1, 2, 3] ; b_list = ["test", "abcd"]

# a_list에 b_list를 연결하세요.
a_list += b_list
print("extend = + 연산 :",a_list)

# a_list 에 b_list를 append를 추가하세요.
a_list = [1, 2, 3]
a_list.append(b_list)
print("append 연산 :",a_list)

# 추가된 a_list에서 "abcd"를 출력하세요 -> index 이용
print(a_list [a_list.index(b_list)] [b_list.index("abcd")] )
#print(a_list[a_list.index("abcd")])

# b_list에서 "test"앞에 "name"을 추가하세요.
b_list.insert(0,"name")
print(b_list)

# a_list의 두번째 요소를 'change'로 변경하세요.
a_list[1] = "change"

# 마지막으로  a_list와 b_list를 출력해보세요.
print(a_list,"\n",b_list)


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
