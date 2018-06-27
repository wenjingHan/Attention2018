import os
import numpy as np

N = 100

prime_nums = np.empty([N], dtype=np.int32)
is_prime = np.ones([N], dtype=bool)
prime_index = 0

for num in range(2,N):
    if is_prime[num]:
        prime_nums[prime_index] = num
        prime_index+=1
    j = 0
    while num*prime_nums[j] < N:
        is_prime[num*prime_nums[j]] = False
        if num % prime_nums[j] == 0:
            break
        j += 1

print(prime_nums)

'''
with open('prime_nums.txt', mode='w') as f:
    for i in range(0, prime_index):
        f.write(str(prime_nums[i]) + ' ')
'''
