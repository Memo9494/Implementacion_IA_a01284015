#AND
n = 0
a = 0
x1 = 0
x2 = 1

def fnot(x1):
    if(x1):
        return 0
    return 1
def f_and(x1,x2):
    p1 = 0.5
    p2 = 0.5
    b1 = -0.5
    b2 = -0.5
    return p1*x1 + b1 + p2*x2 + b2

n = f_and(x1,x2)
if(n < 0):
    a = 0
else:
    a = 1    
print(a)

#ORn = 0
a = 0
x1 = 0
x2 = 1
def f2(x1,x2):
    p1 = 0.5
    p2 = 0.5
    b1 = -0.25
    b2 = -0.25
    return p1*x1 + b1 + p2*x2 + b2

n = f2(x1,x2)
if(n < 0):
    a = 0
else:
    a = 1    
print(a)

#XORn = 0
x1 = 1
x2 = 1

def f(x1,x2):
    p1 = 1
    p2 = 0
    b1 = 1
    b2 = 1
    return p1*x1 + b1 + p2*x2 + b2

n = f(x1,x2)
if(n < 0):
    a = 0
else:
    a = 1    
print(a)


#XOR

a1 = 0
a2 = 0
x1 = 0
x2 = 0

n1 = f_and(fnot(x1),x2)
n2 = f_and(x1,fnot(x2))

if(n1 < 0):
    a1 = 0
else:
    a1 = 1    
print(a1,"a1")

if(n2 < 0):
    a2 = 0
else:
    a2 = 1    
print(a2,"a2")

a3 = 0

n3 = f2(a1,a2)
print(n3)
if(n3 < 0):
    a3 = 0
else:
    a3 = 1

print(a3)