student={'name':'manu','age':20,'dept':'cse'}
print(student)
print(student.keys())
print(student.values())
print(student.keys)
print(student.items())
student.setdefault('is_alive',True)
print(student)
print(student.get('marks','not found'))
print(student.get('marks'))