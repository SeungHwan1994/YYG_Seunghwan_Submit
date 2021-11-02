#!/usr/bin/env python
# coding: utf-8

# In[10]:


# 피보나치 수열 함수 정의
#비 메모화 방법. 시간이 오래걸림.

def fibo(n):
    global counter
    counter += 1
    if n == 1:
        return 1
    if n == 2:
        return 1
    else:
        return fibo(n-1) + fibo(n-2)


# In[2]:


count_fibo = 0

def fibo_m(n,dict_fibo = {1 : 1, 2 : 1}):
    global count_fibo
    count_fibo += 1
    
    if n in dict_fibo:
        return dict_fibo[n]
    else:
        dict_fibo[n] = fibo_m(n-1) + fibo_m(n-2)
        return dict_fibo[n]

print("피보나치 값:",fibo_m(100))
print("횟수 :",count_fibo)


# In[3]:


count_fibo = 0

def fibo_m(n,dict_fibo = {}):
    global count_fibo
    count_fibo += 1
    
    if n == 1:
        return 1
    
    if n == 2:
        return 1
    
    if n in dict_fibo:
        return dict_fibo[n]
    else:
        dict_fibo[n] = fibo_m(n-1) + fibo_m(n-2)
        return dict_fibo[n]

print("피보나치 값:",fibo_m(100))
print("횟수 :",count_fibo)


# In[23]:


counter = 0

print(fibo(10))
print("fibo func {} 계산에 활용된 덧셈 횟수는 {}".format(10,counter))


# In[5]:


dict_fibo= { 1:1 , 2:1 } # 딕셔너리에 재귀함수 값을 저장함.

def fibo_m(n) :
    global count
    count += 1
    
    if n in dict_fibo: # 딕셔너리에 재귀함수 값이 있을 경우, 그대로 출력한다.
        return dict_fibo[n]
    else: # 딕셔너리에 재귀함수 값이 없으면, 계산하고 딕셔너리에 저장한다.
        output = fibo_m(n-1) + fibo_m(n-2)   
        dict_fibo[n] = output # 딕셔너리에 값 저장함.
        print("{} : {}".format(n,output))
        return output


# In[6]:


count = 0

print(fibo_m(10))
print("fibo func {} 계산에 활용된 덧셈 횟수는 {}".format(10,count))


# In[60]:


# tuple : 값 변경 안됨 ( )

tuple_var = (1,2,3)
print(tuple_var[0])

a,b = 10, 20
a,b = b,a

print("a = {}, b = {}".format(a,b))


c, d = (30, 40) # 각각의 요소에 대한 지정이므로, c / d 는 튜플이 아니라 요소임.
c = (10,10)  # c 는 요소가 아니라 튜플 자체로 넣은 것이므로, 튜플임.
print(type(c))

c = (20, 10, 30) # 요소가 아닌 튜플 자체를 바꾸는 것은 변경 가능함.
c[0] = 20 # 이것은 튜플이기때문에 요소를 변경할 수 없음.

print("c = {}, d = {}".format(c,d))


# In[65]:


# lambda 는 함수를 간단하게 정의
# lambda 매개변수 : 리턴값

def power(x):
    return x*x
def under_3(x):
    return x < 3

print("map() : ", list(map(power,[1,2,3,4,5])))
print("filter() : ", list(filter(under_3,[1,2,3,4,5])))


# In[66]:


a_power = lambda x : x * x
a_under = lambda x : x < 3

print("map() : ", list(map(a_power,[1,2,3,4,5])))
print("filter() : ", list(filter(a_under,[1,2,3,4,5])))


# In[67]:


print("map() : ", list(map(lambda x : x * x,[1,2,3,4,5])))
print("filter() : ", list(filter(lambda x : x < 3,[1,2,3,4,5])))


# In[73]:


x, y = 10, 3
a = lambda x, y : x*y

print(a(x,y))


# In[82]:


# file open, write, close
# file = open('경로명/파일명',mode: 'w, r, a')
# 현재 작업하고 있는 디렉토리  = ./
# 상위 디렉토리 = ../
# 현재경로에서 하위 디텍토리 = =./디렉토리명

file = open('./basic.txt','w') # w: 기존에 존재하면 기존 내용 없애고 새로 생성.
file.write("hello python Programming")
file.close()


# In[3]:


# file read
file = open('./basic.txt','r') # 입력모드로, 존재하지않으면 error
a = file.read()
print(a)
file.close()


# In[5]:


# with 문장종료하면 file.close() 자동실행
with open('./basic.txt','r') as file: # with 문장종료하면 file.close() 자동실행
    contents = file.read()
print(contents)


# In[6]:


# 파일명은 score.txt 로
# 이름, 성적을 입력받아 파일에 저장, 'q'가 입력되면 종료.
# 구분자는 홍길동, 90 (구분자는 "," end = "\n" 으로)

# file.write() => print() 함수와 사용법이 유사. 
# file.write("{},{}\n".format(name,score))
# file 읽어서 화면에 출력하세요.

def write_file(score_file):
    input_str = input("name score >"),split()
    if 'q' in input_str:
        return False
    score_file.write("{},{}\n",format(input_str[0],int(input_str[1]))
    return True

with open("./score.txt","w") as file:
    file.write("{}, {}\n".format('name','score'))
    while True:
        if not write_file(score_file):
            break
    score_file.seek(0)
    for line in score_file:
        print(line)
    print("--------------")
    score_file.seek(0)
    print(score.file.read())


# In[2]:


# 파일명은 score.txt 로
# 이름, 성적을 입력받아 파일에 저장, 'q'가 입력되면 종료.
# 구분자는 홍길동, 90 (구분자는 "," end = "\n" 으로)

# file.write() => print() 함수와 사용법이 유사. 
# file.write("{},{}\n".format(name,score))
# file 읽어서 화면에 출력하세요.

def write_file(score_file):
    input_str = input("name score >").split()
    if 'q' in input_str:
        return False
    score_file.write("{},{}\n".format(input_str[0],input_str[1]))
    return True

#w+ : read, write 가능하고 기존파일 내용을 지룸
with open("./score.txt","w+") as score_file: 
    score_file.write("{},{}".format('Name','Score\n'))
    while True:
        if not write_file(score_file):
            break
    score_file.seek(0)
    a = score_file.read()
    print("a = :",a)

    score_file.seek(0) # 포인터를 처음으로 이동시킴
    for i in score_file: # 파일에서 한 라인씩 가져옴
        print(i)
        
    print("--------------")
    score_file.seek(0) # 파일 포인터를 처음으로 이동시킴
    print(score_file.read())


# In[19]:


# 파일에 저장된 자료를 읽어서 딕셔너리에 저장한 후 딕셔너리의 키와 값을 출력

file = open("./score.txt",'r')
dict_score = {}

for line in file:
    line_list = line.split(',')
    print(line_list)
    if line_list[0] == 'Name':
        continue
    dict_score[line_list[0]] = int(line_list[1])
file.close()
print(dict_score)


# In[22]:


with open("./test_1.txt","w+") as file:
    file.write("Helloworld")
    file.write("Helloworld")
    file.write("Helloworld\n")
    file.write("Helloworld")
    file.seek(0)
    print(file.read())
    print("-------------------")
    file.seek(0)
    for i in file:
        print(i)
    


# In[35]:


# yield 키워드를 사용하여 제너레이터 함수 정의

def g_func():
    print('test1')
    yield "test1"
    
    print('test2')
    yield "test2"
    
    print('test3')
    
func = g_func()
print("main ")
g_func() # 제너레이터 함수는 실행되지 않음

# next 함수는 yield까지 실행하고 yield 뒤에 나오는 자료를 return한다.

print(next(func)) # next 함수를 사용해야 실행이 된다.
print("main 1 ---")

print(next(func))
print("main 2 ---")

print(next(func)) # yield 가 없어 return할 값이 없으므로  Error 발생함.
print("main 3 ---")


# In[39]:


# reversed 또한 반복을 수행하는 제너레이터 함수임.

a_list = [1,2,3,4,5]
b_list = reversed(a_list)

print(b_list)
print(next(b_list))
print(next(b_list))
print(next(b_list))
print(next(b_list))
print(next(b_list))
print(next(b_list))
print(next(b_list))


# In[70]:


numbers = [1, 2, 3, 4, 5]
print("::".join(str(i) for i in numbers))

print("::".join(map(str,numbers)))


# In[66]:


numbers = list(range(1, 10 +1))

print("홀수만 추출하기")
print(list(filter(lambda x : x % 2 == 1, numbers)))

print("3 이상, 7미만 추출하기")
print(list(filter(lambda x : 3 <= x < 7, numbers)))

print("제곱해서 50미만 추출하기")
print(list(filter(lambda x : x * x < 50, numbers)))


# In[72]:


# 순서를 바꾸는 이터레이터 함수를 만들고, 바꾼 순서대로 출력해보라.
# 이터레이터 함수가 아닐경우, 아래와 같이 출력된다.

def r_func(v_list):
    for i in range(len(v_list)-1, -1, -1):
        return v_list[i] 

a_list = [10, 20 ,30 ,40, 50]

a = r_func(a_list)

for i in range(len(a_list)):
    print(a)


# In[73]:


# 이터레이터 함수로 만들 경우 아래와 같이 정상작동된다.

def r_func(v_list):
    for i in range(len(v_list)-1, -1, -1):
        yield v_list[i] 

a_list = [10, 20 ,30 ,40, 50]

a = r_func(a_list)

for i in range(len(a_list)):
    print(next(a))


# In[80]:


tuple_a = ()

for i,value in enumerate(['a', 'b', 'c', 'd', 'e']):
    tuple_a += (i,value)
    
print(type(tuple_a))
print(tuple_a)


# In[92]:


# 예외처리 try ~ except ~ else ~ finally
# except Exception as e: Exception() = Error 메시지 출력
pi = 3.14


try:
    number_input = int(input("정수입력 >"))
    print("원의 반지름 :",number_input)
    print("원의 둘레 :",number_input*2*pi)
    print("원의 둘레 :",pi*number_input**2)
except Exception as e: # 에러가 발생할 경우 실행
    print("정수가 입력되지않음")
    print(e)
else: # 에러가 없을 경우 실행
    print("else 이하가 실행됨")
finally: # 무조건 실행
    print("try - except - else - finally")


# In[97]:


# 파일을 r 모드로 오픈한 뒤 에러발생하여 w 모드로 오픈
# 마지막에 close 하고 종료
import os

try:
    file = open('./try.txt','r')
except Exception as e:
    print(e)
    file = open('./try.txt','w')
finally:
    print("file close")
    file.close()
    os.remove('./try.txt')


# In[103]:


import os

print(os.path)


# In[105]:


numbers = [52, 273, 32, 103, 90, 10, 275]

print("(1) 요소 내부에 있는 값 찾기")
print("- {}는 {} 위치에 있습니다.".format(52,numbers.index(52)))
print()

print("(2) 요소 내부에 없는 값 찾기")
number = 10000
try:
    print("- {}는 {} 위치에 있습니다.".format(number,numbers.index(number)))
except:
    print("- 리스트 내부에 없는 값입니다.")
print()

print("정상적으로 종료되었습니다.")


# In[106]:


numbers = [52, 273, 32, 103, 90, 10, 275]

def search_numb(s_num, numbers):
    try:
        print("{}는 {}위치에 있습니다.".format(s_num,numbers.index(s_num)))
    except:
        print("{}는 리스트 내부에 없습니다.".format(s_num))
    finally:
        print("함수가 종료되었습니다.")
        
search_numb(1000,numbers)
search_numb(52,numbers)


# In[117]:


numbers = [52, 273, 32, 103, 90, 10, 275]
try:
    s_num = int(input("정수 입력> ")) # 찾고자하는 인덱스 입력
    print("{}는 {}위치에 있습니다.".format(s_num,numbers[s_num]))
except ValueError as err_contents:
    print("정수를 입력하세요.",type(err_contents))
except IndexError as err_contents:
    print("범위를 벗어났습니다.",type(err_contents))
except Exeption as err_contents:
    print("예측하지 못한 에러 발생.",type(err_contents))


# In[123]:


number = int(input("정수 입력 >"))
if number > 0:
    raise NotImplementedError("아직 구현 안함")
# 강제로 에러를 발생시켜 프로그램 종료
else:
    pass


# In[126]:


while True:  
    try :
        a = int(input("정수입력 >"))
        break
    except:
        print("정수입력하라고")


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
# 
# 
# 
# 
