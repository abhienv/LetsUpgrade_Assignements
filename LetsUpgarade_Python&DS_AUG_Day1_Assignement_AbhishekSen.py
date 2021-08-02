#!/usr/bin/env python
# coding: utf-8

# 1) After running the following code, what does the variable bacon contain?
# 
# bacon = 22
# 
# bacon + 1

# In[2]:


bacon=22
bacon+1 

#variable bacon contains 22


# 2) What should the values of the following two terms be?
# 
# 'spam' + 'spamspam'
# 
# 'spam' * 3
# 

# In[5]:


'spam' + 'spamspam'

'spam' * 3


# 3) How can you tell the difference between break and continue?

# In[ ]:


#'Break' will stop the loop when the condition meets, whereas 'Continue' will skip the result and move forward as per running loop.


# 4) In a for loop, what is the difference between range(10), range(0, 10), and range(0, 10, 1)?

# In[ ]:


#range(10), range (0,10) & range(0,10,1) will return the same values i.e., 0,1,2,3,4,5,6,7,8,9


# 5) Using a for loop, write a short programme that prints the numbers 1 to 10 Then, using a while loop, create an identical programme that prints the numbers 1 to 10.

# In[ ]:


for i in range(1,11):
    print(i)


# In[ ]:


a=1
while a<11:
    print (a)
    a+=1


# 6) Given a number x, determine whether the given number is Armstrong number or not.
# 
# Input : 153
# 
# Output : Yes
# 
# 153 is an Armstrong number.
# 
# 1 * 1 * 1 + 5 * 5 * 5 + 3 * 3 * 3 = 153

# In[ ]:


def power(x, y):
      
    if y == 0:
        return 1
    if y % 2 == 0:
        return power(x, y // 2) * power(x, y // 2)
          
    return x * power(x, y // 2) * power(x, y // 2)
  
def order(x):
    n = 0
    while (x != 0):
        n = n + 1
        x = x // 10
          
    return n
  
def isArmstrong(x):
      
    n = order(x)
    temp = x
    sum1 = 0
              
    while (temp != 0):
        r = temp % 10
        sum1 = sum1 + power(r, n)
        temp = temp // 10
  
    return (sum1 == x)

#print (isArmstrong(153))
#print (isArmstrong(135))


# 7) Program to find Sum of squares of first n natural numbers.
# 

# In[ ]:


def squaresum(n) :
    sm = 0
    for i in range(1, n+1) :
        sm = sm + (i * i)    
    return sm
print(squaresum(n))


# 8) Program to Reverse words in a given String in Python.

# In[ ]:


def rev_sentence(sentence): 
  
    words = sentence.split(' ') 
    reverse_sentence = ' '.join(reversed(words)) 
  
    return reverse_sentence 
  
if __name__ == "__main__": 
    input = 'Learning in LetsUpgrade is Great'
    print (rev_sentence(input))


# 9) Given a list of numbers, write a Python program to find the sum of all the elements in the list.
# 
# Input: [10,12,13]
# 
# Output: 35

# In[ ]:


output = 0
input = [10, 12, 13]
for i in range(0, len(input)):
    output = output + input[i]
print("Output:", output)


# 10) Write a Python program to print all even numbers between 10-1000.

# In[ ]:


for i in range(10,1000,2):
    print(i)

