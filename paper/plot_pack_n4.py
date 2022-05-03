import matplotlib.pyplot as plt

n3_5 = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1,  1.5, 2,  3, 4, 5, 6, 7, 8, 10, 11, 12, 13]
pk_5 = [52,  41,  35,  29,  25,  22,  20,  18, 13,  12, 9, 7, 6, 5, 5, 4, 3,  0,  0,  0]

n3_10 = [3,  4,  5,  6,  7,  8,  9,  10, 11, 13, 15, 17.5, 20, 22.5, 25, 30, 40, 41, 43, 45]
pk_10 = [76, 56, 52, 46, 45, 42, 38, 37, 34, 32, 32, 30  , 29, 27,   24, 23, 21, 0,  0,  0]

n3_20 = [50,  60,  70,  80,  90,  100, 110, 120, 130, 140, 150, 160, 170]
pk_20 = [137, 130, 121, 114, 110, 108, 104, 98,  95,  95,  91,  88,  85]

plt.plot(n3_5,  pk_5,  'r', label =  "5 nodos")
plt.plot(n3_10, pk_10, 'g', label = "10 nodos")
plt.plot(n3_20, pk_20, 'b', label = "20 nodos")     
plt.legend()       
plt.show()