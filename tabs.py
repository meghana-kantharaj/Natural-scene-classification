matrix=[
[' ','coast','forest','mountain','tallbuilding'],
['coast',2923,138,562,634],
['forest',84,3119,323,313],
['mountain',1139,918,1472,934],
['tallBuilding',591,1069,238,2363]
]
summ=0
for i in range(1,5):
    for j in range(1,5):
        summ= summ+matrix[i][j]
print(summ)
tp=[]
tp.append(matrix[1][1]/(matrix[1][1]+matrix[3][3]+matrix[4][4]+matrix[2][2]))
tp.append(matrix[2][2]/(matrix[1][1]+matrix[3][3]+matrix[4][4]+matrix[2][2]))
tp.append(matrix[3][3]/(matrix[1][1]+matrix[3][3]+matrix[4][4]+matrix[2][2]))
tp.append(matrix[4][4]/(matrix[1][1]+matrix[3][3]+matrix[4][4]+matrix[2][2]))
fp=[]
fp.append((matrix[1][3]+matrix[1][4]+matrix[1][2])/(matrix[1][3]+matrix[1][4]+matrix[1][2]+matrix[3][3]+matrix[4][4]+matrix[2][2]))
fp.append((matrix[2][1]+matrix[2][3]+matrix[2][4])/(matrix[2][1]+matrix[2][3]+matrix[2][4]+matrix[1][1]+matrix[3][3]+matrix[4][4]))
fp.append((matrix[3][1]+matrix[3][2]+matrix[3][4])/(matrix[3][1]+matrix[3][2]+matrix[3][4]+matrix[1][1]+matrix[4][4]+matrix[2][2]))
fp.append((matrix[4][1]+matrix[4][2]+matrix[4][3])/(matrix[4][1]+matrix[4][2]+matrix[4][3]+matrix[1][1]+matrix[3][3]+matrix[2][2]))

print(tp)
print(fp)
import matplotlib.pyplot as plt
plt.plot(tp,fp , 'ro')
plt.show()