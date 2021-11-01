#!/usr/bin/env python
# coding: utf-8

# In[19]:


# 리스트 요소 추가, 제거, 변경

a_list = [1,2,3]
b_list = [4,5,6]

# list 요소 추가
a_list.append('abcd')
print("a_list append : ",a_list)
a_list.insert(1,10)
print("a_list insert : ",a_list)

# list에 list 추가
a_list.extend(b_list)
print("a_list extend : ",a_list)

# list 요소 변경
a_list[3] = '변경'
print("a_list 변경 : ",a_list)

# list 요소 삭제
del b_list[1]
print("b_list del : ",b_list)

a_list.pop(2)
print("a_list pop : ",a_list)

a_list.remove('abcd')
print("a_list remove : ",a_list)
a_list.clear()
print("a_list clear : ",a_list)

# list 내부에 요소가 있는지 확인
a_list = [1,2,3,4,5,6,7,8,9,10,[1,2]]
print("6 in a_list :",6 in a_list)
print("11 in a_list :",11 in a_list)
print("1,2 in a_list :",[1,2] in a_list)
print("13 not in a_list :",13 not in a_list)


# In[24]:


# for 반복분 : for 변수 in 반복자료: tab키 들여스기 후 처리문
for i in range(5):
    print("i: ",i)

for i in b_list:
    print("b_list : {}".format(i))
    
for char in "hello python":
    print("char : {}".format(char))
    


# In[26]:


list_of_list =[[1,2,3],[4,5,6,7],[8,9]] # 1 2 3 4 5 6 7 8 9

for i in list_of_list:
    for j in i:
        print(j)


# In[33]:


# 딕셔너리 선언
dictionary = {}

print("요소추가 이전:",dictionary)

#딕셔너리에 요소를 추가합니다.
dictionary["name"] = "새로운 이름"
dictionary["head"] = "새로운 정신"
dictionary["body"] = "새로운 몸","두번째 몸"

print("요소추가 이후:",dictionary)


# In[38]:


# dictionary : { 키 : 값, ... }
# 접근 dic_a[키]

dic_a = {
    "name" : "70 건조 망고",
    "type" : "당절임",
    "ingredient" : ["망고","설탕","메타줌아황산나트륨","치자황색소"],
    "origin" : "필리핀"
}

#사용자로부터 입력을 받습니다.
key = input("> 접근하고자 하는 키: ")

if key in dic_a:
    print(dic_a[key])
else:
    print("존재하지 않는 키에 접근하고 있습니다.")
    
value = dic_a.get(key)
print("키 값 : ",value)
if value == None:
    print("존재하지 않는 키에 접근하고 있습니다.")


# In[60]:


dict_a = {
    "name" : "70 건조 망고",
    "type" : "당절임",
    "ingredient" : ["망고","설탕","메타줌아황산나트륨","치자황색소"],
    "origin" : "필리핀",
    "items" : {"key1" : "dict test", "value" : "dic value"}
}

# for 문장을 이용하여 딕셔너리의 값을 출력하세요.
for i in dict_a:
    print(dict_a[i])
print(type(dict_a), type(dict_a["type"]), type(dict_a["ingredient"]))

# 각 값들만 출력하세요.
for key in dict_a:
    if type(dict_a[key]) == dict:
        for i in dict_a[key]:
             print("dict :",dict_a[key][i])
    if type(dict_a[key]) == list:
        for i in dict_a[key]:
             print("list :",i)
    else:
        print("str :",dict_a[key])

        


# In[73]:


# range(start_num, end_num, step_num) : start_num ~ end_num 까지 step_num 만큼
# 띄어서 숫자를 반복함.

array = [273, 45, 103, 54, 101]
for i in range(len(array)):
    print("{}번째 데이터 : {}".format(i,array[i]))

#뒤에서부터 출력하기
print("----------------------------------")

for i in range(len(array)-1,0 -1,-1):
    print("{}번째 데이터 : {}".format(i,array[i]))

print("----------------------------------")

for i in reversed(range(len(array))):
    print("{}번째 데이터 : {}".format(i,array[i]))


# In[4]:


# while 표현식 : 표현식이 True이면 실행
i = 0
array = [273, 45, 103, 54, 101]

while i < len(array):
    print("{}번째 데이터 : {}".format(i,array[i]))
    i += 1


# In[3]:


# 빈 리스트를 하나 생성한 후 키보에서 자료를 입력받아 리스트에 추가
# 입력 값이 'quit'이면 입력 종료 후 입력된 list의 각각의 값을 추가

list_c = []
a = input("자료를 입력하세요 :")

while 'quit' not in list_c:
    a = input("자료를 입력하세요 :")
    list_c.append(a)

for i in list_c:
            print(i)


# In[10]:


##### input_data = []
input_db = []
input_data = []

while True:
    input_str = input("문자 입력 :")
    input_db.append(input_str)
    
    if input_str.isdecimal():
        continue # while True로 이동
    
    if input_str == 'quit':
        break
    
    input_data.append(input_str)
    
#출력
for char in input_data:
    print(char)
    
print("input data :",input_db)


# In[12]:


# min(a) , max(a) , sum(a) -> a 는 리스트 형식이어야함

a_list = [34, 54, 78, 100, 320, 43]

print("min : {}, max : {}, sum : {}".
      format(min(a_list),max(a_list),sum(a_list)))


# In[16]:


# reversed() : 리스트 뒤집기

print("a_list : {}, \na_list_reversed : {}".
     format(a_list,list(reversed(a_list))))


# In[20]:


# enumerate() : 인덱스와 값을 튜플형태로 반환함.
for i,value in enumerate(reversed(a_list)):
    print("{}번째요소 : {}".format(i,value))


# In[22]:


# enumerate() : 인덱스와 값을 튜플형태로 반환함.
for i in enumerate(reversed(a_list)):
    print(i)


# In[27]:


dict_a = {"name" : "A B C D","Number" : "1 2 3 4"}

for i in dict_a:
    print(i)

for i in dict_a.items():
    print(i)
    
for i in enumerate(dict_a):
    print(i)


# In[ ]:


# 키보드에서 이름을 입력받아 리스트에 저장한 후 ( 입력은 이름에 'q'가 들어가면 종료)
# 검색하고자하는 이름을 입력받아 몇번째 입력된 이름인지 출력하세요.
# enumerate 함수를 이용하세요.

input_name = []

while True:
    input_str = input("이름 입력 (q = 종료) >")
    if input_str == 'q':
        break
    input_name.append(input_str)

search_name = input("검색할 이름>")

for idx, name in enumerate(input_name):
    if name == search_name:
        print("{}번째 데이터는 {} 입니다.".format(idx+1,name))
        break


# In[5]:


# 입력한 이름의 객수만큼 점수를 입력하여 score[] 리스트에 저장하세요.
# 저장된 이름과 점수를 출력 : 홀길동 90 김철수 100
# 딕셔너리로 변환해서 출력
input_name = []

while True:
    input_str = input("이름 입력 (끝 = 종료) >")
    if input_str == '끝':
        break
    input_name.append(input_str)


score = [] # 라스트 선언
i = 0

while i < len(input_name):
    score.append(int(input("점수입력 :")))
    i += 1
    
print(score)

for i in range(len(input_name)): # 이름과 점수를 출력
    print("{} : {}".format(input_name[i], score[i]))
    

    
dict_score = {} # 딕셔너리 선언
i = 0

while i < len(input_name): # 딕셔너리로 변환
    dict_score[input_name[i]] = score[i]
    i += 1
    
print(dict_score)

for key, value in dict_score.items(): # 출력
    print("{} : {}".format(key,value))


# In[32]:


# 리스트 내포 [ 표현식 for 반복자 in 반복자료 (if 조건문) ]
array = []
for i in range(0,20,2):
    array.append(i * i)
print(array)

print("-----------------------------------")

array_in = [i * i for i in range(0,20,2)]
print(array_in)
print("-----------------------------------")

array_1 = [ i for i in range(100) if i%5 == 0 ] # 5의 배수만 리스트에 포함
print(array_1)
print("-----------------------------------")

# 키보드에서 자료를 입력받아 한글자씩 리스트에 저장한 후
# 문자만 리스트로 출력

input_str = list(input("자료 입력>"))
array_00 = [char for char in input_str if char.isalpha() ]
print(array_00)

# 아래와 동일함
array_00 = []

for i in input_str:
    if i.isalpha():
        array_00.append(i)
print(array_00)


# In[48]:


# 문자열.join (문자열로 구성돈 리스트)
join_char = ":".join(["1","2","3"]) # 리스트를 하나의 문자열로 병합
print(join_char)

# 이터레이터
numbers = [1, 2, 3, 4, 5]

r_num = reversed(numbers) # r_num 는 위치만 저장, 데이터ㅡㄹ 가져오는 방법은 next 함수

next(r_num)

print('first')

for i in r_num:
    print(i)

print('second')
for i in r_num:
    print(i)


# In[70]:


# def 함수명() : 매개변수 return

def print_func(a=10, b=20):
    print("function test")
    return (a,b)

def print_n_times(value, n):
    for i in range(n):
        print(value)

def print_times(value, n=3): # 기본매개변수 
    for i in range(n):
        print(value)
        
def print_var_times(n, *values): # 가변매개변수 
    for i in range(n):
        for value in values:
            print(value)
        
        
def print_var_basic_times(*values, n=3): # 가변매개변수 
    for i in range(n):
        for value in values:
            print(value)
        
def sum_func(start=0, stop=100, step=1):
    tot = 0
    for i in range(start,stop+1,step):
        tot += i
    return tot
        
print(sum_func())
print(sum_func(stop=150))
        
print(print_func())
print_n_times("test",3)
print_times("test123")
print_var_times(2, "abcd", 'test', 123)
print_var_basic_times(2, "abcd", 'test',n=1)


# In[22]:


# 키보드에서 입력받는 함수 : 숫자와 부호를 입력받아 list로 반환하는 함수
# 입력의 끝은 'q' 종료
# 부호가 '+ 이면 합을 구하는 함수'
# 부호가 '- 이면 차을 구하는 함수
# 부호가 '* 이면 곱을 구하는 함수
# 부호가 '% 이면 나눔을 구하는 함수

def plus_func(num1,num2):
    return int(num1) + int(num2)

def minus_func(num1,num2):
    return int(num1) - int(num2)

def input_data(value):
    input_list = value.split()
    
    if input_list[1] == "+":
        result = plus_func(input_list[0],input_list[2])
    elif input_list[1] == "-":
        result = minus_func(input_list[0],input_list[2])
        
    print("{} {} {} = {}".format(input_list[0],input_list[1],input_list[2],
                                result))
    
while True:
    a = input("입력>")
    if a == "q":
        break
    input_data(a)


# In[25]:


#이름과 성적을 입력받아 딕셔너리에 저장한 후 이름을 키로 성적을 값으로
# 이름에 'q'가 입력되면 입력 종료

def input_data():
    input_str = input("name score > ").split()
    if 'q' in input_str:
        return False
    dict_score[input_str[0]] = input_str[1]
    return True

dict_score = {}

while True:
    if not input_data():
        break
        
print(dict_score)


# In[ ]:


























