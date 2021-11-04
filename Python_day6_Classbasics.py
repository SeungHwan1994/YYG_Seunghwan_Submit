#!/usr/bin/env python
# coding: utf-8

# In[9]:


# 학생 리스트를 선언합니다.
students = [
    { "name": "윤인성", "korean": 87, "math": 98, "english": 88, "science": 95 },
    { "name": "연하진", "korean": 92, "math": 98, "english": 96, "science": 98 },
    { "name": "구지연", "korean": 76, "math": 96, "english": 94, "science": 90 },
    { "name": "나선주", "korean": 98, "math": 92, "english": 96, "science": 92 },
    { "name": "윤아린", "korean": 95, "math": 98, "english": 98, "science": 98 },
    { "name": "윤명월", "korean": 64, "math": 88, "english": 92, "science": 92 }
]


# In[10]:


# 학생의 점수 출력

print("이름","국어","수학","영어","과학",sep="\t")
print("====","====","====","====","====",sep="\t")

for student in students:
    for student_data in student:
        print(student[student_data],end="\t")
    print()


# In[22]:


# 학생의 점수 출력

print("이름","국어","수학","영어","과학",sep="\t")
print("====","====","====","====","====",sep="\t")

for student in students:
#    print(student['name'],student['korean'],student['math'],student['english'],
#          student['science'],sep="\t")
    
    print("{}\t{:3d}\t{:3d}\t{:3d}\t{:3d}".
          format(student['name'],student['korean'],student['math'],student['english'],student['science'])) # 자리 똑바르게 출력


# In[34]:


# 학생의 점수 출력

print("이름","총점","평균","국어","수학","영어","과학",sep="\t")
print("====","====","====","====","====","====","====",sep="\t")

for student in students:
    tot = student['korean'] + student['math'] + student['english'] + student['science']
    totnorm = tot / 4
    
    print("{}\t{:3d}\t{:3.1f}\t{:3d}\t{:3d}\t{:3d}\t{:3d}".
          format(student['name'],tot,totnorm,student['korean'],student['math'],student['english'],student['science'])) # 자리 똑바르게 출력


# In[22]:


# 학생 객체를 만든 함수로 학생 자료 생성

def create_student(name,korean,math,english,science):
    return {"name" : name, "korean" : korean, "math" : math,
            "english" : english, "science" : science}

def print_student(students):
    print("이름","총점","평균","국어","수학","영어","과학",sep="\t")
    print("====","====","====","====","====","====","====",sep="\t")
    
    for student in students:
        print(print_string(student))

def total_score(student):
    return student['korean'] + student['math'] + student['english'] + student['science']

def avg_score(student):
    return total_score(student)/4

def print_string(student):
    return "{}\t{:3d}\t{:3.1f}\t{:3d}\t{:3d}\t{:3d}\t{:3d}".format(
    student['name'],total_score(student),avg_score(student),student['korean'],
                     student['math'],student['english'],
                     student['science'])

# 학생 리스트 선언

student = [create_student("윤인성",87,98,88,95),
           create_student("연하진",92,98,96,98),
           create_student("구지연",76,96,94,90),
           create_student("나선주",98,92,96,92),
           create_student("윤아린",95,98,98,98),
           create_student("윤명월",64,88,92,92)
          ]

print_student(student)


# In[19]:


# 학생 객체를 만든 함수로 학생 자료 생성
def create_student(name,korean,math,english,science):
    return {"name": name, "korean": korean, 
            "math": math, "english": english, 
            "science": science
           }

# 총점 구하는 함수
def total_score(student):
    return student['korean'] + student['english'] + student['math']+student['science']
# 평균 구하는 함수
def avg_score(student):
    return total_score(student)/4
#각각의 학생의정보를 출력하는 함수
def print_string(student):
    return "{}\t{}\t{}\t{}\t{}\t{}\t{}".format(student['name'],
                                               total_score(student),
                                               avg_score(student), 
                                               student['korean'],student['english'],
                                               student['math'],student['science'])
students = [
    create_student("윤인성", 87, 98, 88, 95),
    create_student("연하진", 92, 98, 96, 98),
    create_student("구지연", 76, 96, 94, 90),
    create_student("나선주", 98, 92, 96, 92),
    create_student("윤아린", 95, 98, 98, 98),
    create_student("윤명월", 64, 88, 92, 92)
]

print(" 이름","총점","평균","국어","영어","수학","과학",sep='\t')
print("======","====","====","====","====","====","====",sep='\t')
for student in students:
    print(print_string(student))


# In[3]:


# 클래스 객체를 조금 더 효율적으로 생성하기 위해 만들어진 구문
# class로 학생정보 생성 class = 틀

class Student: # 학생 인스턴스에 대한 속성과 메서드를 정의함.
    # 초기화 class 생성시 매개변수들을 정의하고, 클래스 내부의 변수를 정의함.
    def __init__(self,name,korean,math,english,science):
        self.name = name # 클래스의 변수가 지정된다.
        self.korean = korean
        self.math = math
        self.english = english
        self.science = science
    
    #총점 계산 메서드
    def get_sum(self):
        return self.korean + self.math + self.english + self.science
    #평균 계산 메서드
    def get_avg(self):
        return self.get_sum()/4
    #출력 메서드
    def to_string(self):
        return "{}\t{}\t{}\t{}\t{}\t{}\t{}".format(self.name,
                                               self.get_sum(),
                                               self.get_avg(), 
                                               self.korean,self.math,
                                               self.english,self.science)

students_class = [
    Student("윤인성", 87, 98, 88, 95),
    Student("연하진", 92, 98, 96, 98),
    Student("구지연", 76, 96, 94, 90),
    Student("나선주", 98, 92, 96, 92),
    Student("윤아린", 95, 98, 98, 98),
    Student("윤명월", 64, 88, 92, 92)
]

for student in students_class:
    print(student.to_string())


# In[7]:


# 합계와 평균 구하는 함수를 만들어라.
# 가변 매개함수는 모든 매개함수가 리스트로 저장된다.

def f_total(num1,*num2):
    for i in num2:
        num1 += i
    return num1

def f_avg(num1,*num2):
    return f_total(num1,*num2) / (len(num2)+1)

f_total(10,10,10,10,10)
f_avg(10,20,30,20,25,10,50,24)


# In[ ]:


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


# In[18]:


# class 에는 입력, 저장하는 함수 포함
class ClassScoreList:
    def __init__(self,class_name):
        self.class_name = class_name
        self.list_name = []
        self.list_score = []
        self.dict_score = {}
        
    def input_to_dict(self):
        while True:
            input_data = input("(종료) name >")
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
    
ClassScoreList.input_to_dict()


# In[ ]:


# 이름, 국어, 영어 성적을 입력해서 파일에 저장하는 프로그램 작성
# 이름, 국어, 영어 성적을 멤버로 하는 클래스 생성
# 자료를 저장하는 class내의 메서드 정의
# 기존의 파일에서 자료를 가져와서 클래스를 만들고
# 검색해서 성적을 출력하는 프로그램 작성하세요.


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
