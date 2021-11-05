#!/usr/bin/env python
# coding: utf-8

# In[7]:


# class 에는 입력, 저장하는 함수 포함

class ClassScoreList:
    def __init__(self,class_name):
        self.class_name = class_name
        self.list_name = []
        self.list_score = []
        self.dict_score = {}
        
    def input_to_dict(self):
        while True:
            input_data = input(self.class_name + " name (종료) >")
            if input_data == '종료':
                break
            self.list_name.append(input_data)
            
        while True:
            if len(self.list_score) >= len(self.list_name):
                break
            input_data = input("score >")
            self.list_score.append(int(input_data))
            
        for i in range(len(self.list_name)):
            key = self.list_name[i]
            value = self.list_score[i]
            self.dict_score[key] = value
            
        return self.dict_score
    
    def dict_to_file(self):
        file  = open('./file_score_{}.txt'.format(self.class_name),'w+')
        file.write(self.class_name + "\n")
        
        for key,value in self.dict_score.items():
            file.write("{},{}\n".format(key,value))
            
        
        file.seek(0)
        print(file.read())
        file.close()
        
    def file_to_dict(self):
        file = open('./file_score_{}.txt'.format(self.class_name),'r')
        file.seek(0)
        
        for line in file:
            list_score = line.split(',')
            
            if len(list_score) == 1:
                pass
            else:
                self.dict_score[list_score[0]] = int(list_score[1])
            
        return self.dict_score


# In[11]:


# class 정릐, class 생성
# class 클래스명:
# 클래스명() -> 생성자 -> instance

class Aclass:
    def __init__(self,num1,num2):
        self.num1 = num1
        self.num2 = num2
        print("class 생성")
    
    def get_sum(self):
        print("class 내부 함수 실행")
        return (self.num1 + self.num2)
    
    # 파이썬에서 실행하면, 프로그램이 종료될때 클래스가 자동으로 소멸된다.
    def __del__(self): 
        print("class 소멸")

a = Aclass(10,20)
print(a.num1, a.num2, a.get_sum())

b = Aclass(30,40)
print(b.num1, b.num2, b.get_sum())

a_list = [a,b]


# In[14]:


class Student:
    def __init__(self, name, kor, eng):
        self.name = name
        self.kor = kor
        self.eng = eng
    
    def file_write(self):
        fp.write(self.name + ','+ str(self.kor) +',' + str(self.eng)+'\n')
        
def input_data():
    name = input("이름 : ")
    if name == 'q':
        return 'q'
    while True:
        score = input("국어, 영어 점수 입력 : ( 50 40)").split() 
        if len(score) == 2:
            break

    student = Student(name, int(score[0]), int(score[1]))
    student.file_write()
    return
    
def search_data():
    fp.seek(0)
    s_name = input("검색할 이름 입력 ")
    for line in fp:
        line_data = line.split(',')
        if s_name == line_data[0]:
            return line_data
    return 


# In[ ]:


fp = open("class_file.txt","a+")
try:
    while input_data() != 'q':  # 'q' 리턴되면 종료
        continue
    data = search_data()     
    if data:
        print("{} : {}, {}".format(data[0],data[1],data[2]))
    else:
        print("검색할 자료 없음")
              
except Exception as e:
    print("error", e)
finally:
    fp.close()


# In[13]:


# 학생 클래스를 선언합니다.
class Student:
    def study(self):
        print("공부를 합니다.")

# 선생님 클래스를 선언합니다.
class Teacher:
    def teach(self):
        print("학생을 가르칩니다.")

# 교실 내부의 객체 리스트를 생성합니다.
classroom = [Student(), Student(), Teacher(), Student(), Student()]

# 반복을 적용해서 적절한 함수를 호출하게 합니다.
for person in classroom:
    if isinstance(person, Student):
        person.study()
    elif isinstance(person, Teacher):
        person.teach()


# In[ ]:


students = [
    { "name": "윤인성", "korean": 87, "math": 98, "english": 88, "science": 95 },
    { "name": "연하진", "korean": 92, "math": 98, "english": 96, "science": 98 },
    { "name": "구지연", "korean": 76, "math": 96, "english": 94, "science": 90 },
    { "name": "나선주", "korean": 98, "math": 92, "english": 96, "science": 92 },
    { "name": "윤아린", "korean": 95, "math": 98, "english": 98, "science": 98 },
    { "name": "윤명월", "korean": 64, "math": 88, "english": 92, "science": 92 }
]

for student in 


# In[7]:


class Stu:
    count = 0
    
    def __init__(self, name , score):
        self.name = name
        self.score = score
        
        Stu.count += 1
        print("{}번째 입력".format(Stu.count))

stu_list = (Stu('a',90),Stu('b',80))
Stu('a',90)

print("클래스변수 :",Stu.count)


# In[11]:


# 이름과 성적을 입력받아 Student 클래스의 인스턴스에 저장 : lsit (클래스의 리스트)
# 이름에 'q' 입력되면 입력 종료
# 입력된 숫자만큼 데이터 출력

class Student:
    count = 0
    
    def __init__(self, name, score):
        self.name = name
        self.score = score
        
        Student.count += 1
        
students = []
while True:
    name = input("이름 입력 >")
    if name == 'q':
        break
    score = int(input())
    students.append(Student(name,score))
    
print("입력한 총 개수 : {}".format(Student.count))

for i in range(Student.count):
    print(students[i].name,students[i].score)


# In[9]:


# 모듈을 가져옵니다.
import math

# 클래스를 선언합니다.
class Circle:
    def __init__(self, radius):
        self.__radius = radius
    def get_circumference(self):
        return 2 * math.pi * self.__radius
    def get_area(self):
        return math.pi * (self.__radius ** 2)

# 원의 둘레와 넓이를 구합니다.
circle = Circle(10)

print("원의 둘레:", circle.get_circumference())
print("원의 넓이:", circle.get_area())

circle.__radius = 100 # radius의 값이 변경되지않는다.

print("원의 둘레:", circle.get_circumference())
print("원의 넓이:", circle.get_area())


# In[10]:


# 모듈을 가져옵니다.
import math

# 클래스를 선언합니다.
class Circle:
    def __init__(self, radius):
        self.__radius = radius
    def get_circumference(self):
        return 2 * math.pi * self.__radius
    def get_area(self):
        return math.pi * (self.__radius ** 2)
    
    def get_radius(self):
        return self.__radius
    
    def set_radius(self,value):
        self.__radius = value
        

# 원의 둘레와 넓이를 구합니다.
circle = Circle(10)
print("원의 둘레:", circle.get_circumference())
print("원의 넓이:", circle.get_area())

circle.set_radius(15) # radius가 15로 변경된다.

print(circle.get_radius())
print("원의 둘레:", circle.get_circumference())
print("원의 넓이:", circle.get_area())


# In[19]:


# 상속

class Parent:
    def __init__(self):
        self.value = '테스트'
        print("Parent 클래스의 __init__ 메서드가 호출되었습니다.")
    def test(self):
        print("parent 클래스의 test() 메서드입니다.")
        
# 자식 클래스를 선언합니다.
class Child(Parent):
    def __init__(self):
        Parent.__init__(self)
        print("Child 클래스의 __init()__ 메서드가 호출되었습니다.")
    def test(self):
        print("Child 클래스의 test() 메서드입니다.")
# 자식 클래스의 인스턴스를 생성하고 부모의 메서드를 호출합니다.

child = Child() # Parent 의 init 과 child 의 init이 같이 실행됨.

child.test() # Child 클래스의 함수가 실행됨.
print("child.value :",child.value) # Parent의 변수를 가져올 수 있음.


# In[2]:


# numpy : 배열의 연산등을 쉽게 처리 가능하도록 하는 패키지
import numpy as np


# In[31]:


a = np.arange(15)
print(a)
print("a.ndim = {}\n,a.shape = {}\n,a.size = {}\n,a.dtype.name = {}\n,a.dtype.itemsize = {}\n".
      format(a.ndim,a.shape,a.size,a.dtype.name,a.dtype.itemsize))
a = np.arange(15).reshape(3,5)
print(a,type(a))
print("a.ndim = {}\n,a.shape = {}\n,a.size = {}\n,a.dtype.name = {}\n,a.dtype.itemsize = {}\n".
      format(a.ndim,a.shape,a.size,a.dtype.name,a.dtype.itemsize))


# In[28]:


np.arange(-10,5,0.5).reshape(5,6)


# In[33]:


np.arange(-10,5,0.5)


# In[38]:


# list 에서 array를 생성하는 방법

a_list = [1,2,3,4,5,6]
a_array = np.array(a_list)
print(type(a_list),type(a_array),"\n",a_list,a_array)


# In[55]:


# inf, Nan, 초기값 설정한 배열 생성 (변수로치면 선언하는 것과 같음)
a = np.zeros((2,3,4)) # 0을 초기값으로 배열을 생성한다.
b = np.ones((2,3,4),dtype='U4') # 1을 초기값으로 배열을 생성한다.
print(a)
print(b)


# In[53]:


a = np.empty((4,3)) # 메모리가 초기화되어있지않아 쓰레기값이 나오게됨.
a


# In[16]:


# array 연산
x1 = np.array([1,2,3,4])
y1 = np.array([4,5,6,7])

x2 = np.array([[1,2],[3,4]])
z1 = np.array([-1,-2])
z2 = np.array([[1,2,3,4],[5,6,7,8]])

print("x1 + y1 :\n",x1 + y1)
print("x2 + z1 :\n",x2 + z1)
print("x2.flatten() :\n",x2.flatten())
print("x1.reshape(2,2) :\n",x1.reshape(2,2))
print("z2.reshape(2,2,2) :\n",z2.reshape(2,2,2))


# In[67]:


# array indexing, slicing, iteraring

a = np.arange(10) ** 2
print(a)
print(a[2])
print(a[3:5])
print(a[0:6:2]) # [start : end : step]
a[0:6:2] = -a[0:6:2]
print(a[0:6:2])
print(a[ : : -1]) # reversed array

print("----------------------------------------------")

b = a.reshape(2,5)
print(b)
print(b[1, : ]) # 2번째 행만 출력
print(b[ : ,1]) # 2번째 열만 출력
print(b[-1,-1]) # 해당 좌표 출력, 마지막 행, 마지막 열


# In[68]:


a = np.array([[2,3,4],[5,6,2],[9,6,7]])
print(a)
print(np.argmax(a, axis=1)) # x축 = 1 / y축 = 0 / z축 = 2 
print(np.argmax(a, axis=0))


# In[74]:


print(np.mean(a, axis=1)) # 행단위로 평균값
print(np.var(a)) # 
print(np.median(a)) # 전체에 대한 중간값
print(len(a)) # 행의 개수
print(a.size)


# In[110]:


np.random.seed(100) #  난수의 값을 고정 시키기위해 사용하는 함수
np.random.rand(3) # 게속 재실행해도 난수의 값은 동일하다.


# In[111]:


# 데이터의 순서바꾸기

x = np.arange(10)
print(x)
np.random.shuffle(x)
print(x)


# In[178]:


# choice 함수 : sampling
# np.random.choice(data,size=None,replace=True,p=None)
# a : 배열이면 원래의 데이터, 정수이면 arange 명령으로 데잍 생성
# size : 정수, 샘플 숫자
# replace : 불리언. True이면 한번 선택한 데이터를 다시 선택가능
# p : 배열. 각 데이터가 선택될 수 있는 확률

# np.random.choice(5,3)  = 5개 중에서 3개 선택
np.random.choice(4,3, p = (0.2,0.5,0.15,0.15))   # 각 확률을 지정하여 선택함.


# In[188]:


np.random.randint(10,size=(2,5)) # 2,5 배열 형태로 0~9까지의 정수난수 생성


# In[209]:


# 실습
# 다음 행렬과 같은 배열이 있다.
x = np.array([range(1,20+1)])
# 이 배열에서 3의 배수를 찾아라.
print(x [x%3 == 0] )
# 이 배열에서 4로 나누면 1이 남는 수를 찾아라
print(x [x%4 == 1] )
# 이 배열에서 3으로 나누면 나누어지고 4로 나누면 1이 남는 수를 찾아라
y = x [ x%3 == 0]
print(y [y%4 == 1])

# 동일한 문법
print(np.intersect1d(x [x%3 == 0], x [x%4 == 1]))

# 동일한 문법
print( x [ (x %3 == 0) & (x %4 == 1) ] ) # and = & / or = |
print( x [ (x %3 == 0) | (x %4 == 1) ] )


# In[208]:


(x %3 == 0) & (x %4 == 1) # True or False 반환

# x [ ] -> [ ] 안에 True인 요소만 출력


# In[226]:


# 1부터 100까지의 수중에서 3의 배수만 배열로 생성
x = np.arange(1,100 + 1)
x = x [ x%3 == 0 ]
print(x)
# 위에서 생성된 배열의 값 중에서 임의로 5개의 값을 출력
print(np.random.choice(x,5))
# 위의 배열의 사이즈를 확인한 후, 행이 3인 2차원 배열을 생성
print(np.size(x))
x_1 = x.reshape(3,11)
print(x_1)
# 배열의 원소중에서 5의 배수만 출력
print(x_1 [ x_1%5 == 0])


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
