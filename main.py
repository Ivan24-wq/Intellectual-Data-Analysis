import math
#Класс для работы с дробью
class Frac:
    def __init__(self, numerator, denominator):
        if denominator == 0:
            raise ValueError("Знаменатель не может равняться нулю!")
        
        self.numerator = numerator
        self.denominator = denominator
        self._reduce()
    
    #Сокращение дробей
    def _reduce(self):
        g = math.gcd(self.numerator, self.denominator)
        self.numerator //= g
        self.denominator //= g
        if self.denominator < 0:
            self.numerator = -self.numerator
            self.denominator = -self.denominator
    #вывод в виде дроби
    def __str__(self):
        if self.denominator == 1:
            return str(self.numerator)
        return f"{self.numerator}/{self.denominator}"
    
    #Функция для возврата обратной дроби
    def inverse(self):
        if self.numerator == 0:
            raise ZeroDivisionError("Нельзя обратить нулевую дробь!")
        return Frac(self.denominator, self.numerator)
    
    #Словжение дробей
    def __add__(self, other):
        if not isinstance(other, Frac):
            other = Frac(other)
        new_numerator = self.numerator * other.denominator + self.denominator * other.numerator
        new_denominator = self.denominator * other.denominator
        return Frac(new_numerator, new_denominator)
    #Умножение дробей
    def __mul__(self, other):
        if not isinstance(other, Frac):
            other = Frac(other)
        new_numerator = self.numerator * other.numerator
        new_denominator = self.denominator * other.denominator
        return Frac(new_numerator, new_denominator)
    

a = Frac(2, 3)
b = Frac(3, 4)
print("a = ", a)
print("b = ", b)
print("a + b = ", a + b)
print("a * b = ", a * b)
print("inverse(a) = ", a.inverse())