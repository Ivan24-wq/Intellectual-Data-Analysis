import pylab

#Функция f1
def f1(x): 
    return 3 *x*x

#Функция f2
def f2(x):
    return -8 * x + 11

#произвольным образом подставляем значения f1 и f2

x =[i for i in range(-100, 100)]
y1 =[f1(i) for i in x]
y2=[f2(i) for i in x]
pylab.plot(x, y1)
pylab.figure(1)
pylab.plot(x, y2)
pylab.show()
