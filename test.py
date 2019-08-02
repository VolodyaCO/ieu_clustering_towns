import time    

start = time.time()

with open("test1.txt", 'w') as f:
    for i in range(10000000):
        print(i)
        f.write('This is a speed test\n')
end = time.time()
print(end - start)
