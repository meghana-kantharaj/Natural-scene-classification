import os
from random import shuffle
import numpy
import pickle
from PIL import Image
dataset={ 'coast':0, 'forest':1,  'mountain':2,  'tallbuilding':3};


images=[]
classes=[]
for category,label in dataset.items():
	
	count=0
	for file in os.listdir('../dataset/'+category):
		count=count+1
		colorImg=Image.open(open('../dataset/'+category+'/'+file,'rb'))
		grayImg=colorImg.convert('L')
		wrapper=[]
		wrapper.append(numpy.array(grayImg))
		wrapper.append(label)
		images.append(numpy.array(wrapper))
		#classes.append(label)
	print(category, ':', count)
shuffle(images)
images2=[]
for im in images :
	classes.append(im[1])
	km=[]
	#im.pop(1)
	km.append(im[0])
	images2.append(km)
images2=numpy.array(images2)
classes= numpy.array(classes)
#print(classes)
print('0000000000000000000000000000000000000000000000000000000000000000000000000000')
print(len(images2),'images');
print(len(classes),'classes');
#print()
print(classes[0],classes[1])
#print(len(classes))

training_x=images2[:2127]
training_y=classes[:2127]
testing_x=images2[2127:]
testing_y=classes[2127:]
"""
img = Image.fromarray(images2[0][0])
import matplotlib.pyplot as plt
plt.imshow(img)
plt.show()
img2= Image.fromarray(images2[1][0])
plt.imshow(img2)
plt.show()
"""
#img.save('my.png')

data = [training_x,training_y ,testing_x,testing_y]
f = open('scenes4.pkl','wb')
pickle.dump(data, f, protocol=2)
f.close()