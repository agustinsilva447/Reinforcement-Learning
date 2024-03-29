import matplotlib.pyplot as plt

n3_5 = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1,  1.5, 2,  3, 4, 5, 6, 7, 8, 10, 11, 12, 13]
pk_5 = [52,  41,  35,  29,  25,  22,  20,  18, 13,  12, 9, 7, 6, 5, 5, 4, 3,  0,  0,  0]

n3_10 = [3,  4,  5,  6,  7,  8,  9,  10, 11, 13, 15, 17.5, 20, 22.5, 25, 30, 40, 41, 43, 45]
pk_10 = [76, 56, 52, 46, 45, 42, 38, 37, 34, 32, 32, 30  , 29, 27,   24, 23, 21, 0,  0,  0]

n3_15 = [6,   8,   10,  15,  20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 81, 85, 90]
pk_15 = [171, 139, 120, 101, 86, 82, 74, 71, 67, 64, 62, 59, 56, 56, 55, 51, 50, 0,  0,  0]

n3_20 = [20,  25,  30,  40,  50,  60,  70,  80,  90,  100, 110, 120, 130, 140, 150, 160, 170, 175, 190, 200]
pk_20 = [200, 184, 168, 150, 137, 130, 121, 114, 110, 108, 104, 98,  95,  95,  91,  88,  85,  0,   0,   0]

nodos = [5,  7,  9,  10, 11, 13, 15, 17,  19,  20]
mxdis = [11, 15, 31, 41, 50, 65, 81, 110, 140, 175]

plt.plot(n3_5,  pk_5,  'r', label =  "5 nodes")
plt.plot(n3_10, pk_10, 'g', label = "10 nodes")
plt.plot(n3_15, pk_15, 'c', label = "15 nodes")
plt.plot(n3_20, pk_20, 'b', label = "20 nodes")    
plt.title("N° of packets for quantum protocol to surpass classical.")
plt.xlabel("Size of Network")
plt.ylabel("Number of packets")        
plt.legend() 
plt.show()

plt.plot(nodos, mxdis, marker='o')
plt.title("Distance for quantum protocol to surpass classical.")
plt.xlabel("N° Nodes")
plt.ylabel("Size of Network")  
plt.show()