from typing import Sized
from keras.backend import zeros
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import shutil
import os
# from numpy.core.defchararray import asarray, count
# from numpy.core.fromnumeric import mean, size
# from numpy.core.records import array
# from numpy.lib.function_base import _parse_input_dimensions
import numpy as np
import scipy.io
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
import csv



'''Stanford Dogs Dataset
http://vision.stanford.edu/aditya86/ImageNetDogs/'''
'''Number of categories: 120
Number of images: 20,580
Annotations: Class labels, Bounding boxes'''


## managing anime dataset
print(os.listdir('Images'))

## generate a ndarray of images's sizes and surfaces
# input: basepath
# output: size's ndarray, surface's ndarray
def generate_listOfSizesAndSurface(basepath):
	#dstDirectory = 'anime'
	#addToName = 'copy_'
	listOfSizes = list()
	listOfSurface = list()
	#getting sizes of anime images and calculating the surface (width*heigth) for each one
	with os.scandir(basepath) as srcDirectories: # inside basepath
		for entry in srcDirectories: # for each object in basepath
			#print(entry.name)
			if entry.is_file():
				#os.rename(basepath+'/'+entry.name,basepath+'/'+addToName+entry.name)
				#shutil.move(basepath+'/'+entry.name, basepath+'/'+dstDirectory)
				img= load_img(basepath+'/'+entry.name)
				img= img_to_array(img)
				#print(img.shape)
				listOfSizes.append(img.shape)
				listOfSurface.append(img.shape[0]*img.shape[1])
	listOfSizes = np.array(listOfSizes)
	listOfSurface = np.array(listOfSurface)
	return listOfSizes, listOfSurface

realSelected_sizeList, realSelected_surfaceList = generate_listOfSizesAndSurface('All_images/real_selected')

## generate a ndarray of images's sizes and surfaces
# input: basepath
# output: size's ndarray, surface's ndarray
def generate_listOfSizesAndSurface(basepath):
	#dstDirectory = 'anime'
	#addToName = 'copy_'
	listOfSizes = list()
	listOfSurface = list()
	#getting sizes of anime images and calculating the surface (width*heigth) for each one
	with os.scandir(basepath) as srcDirectories: # inside basepath
		for entry in srcDirectories: # for each object in basepath
			if entry.is_dir() and str(entry.name)[0] != 'a':
				sizes = list()
				surfaces = list()
				sizes.append(entry.name)
				surfaces.append(entry.name)
				with os.scandir(entry) as subSrcDirectories: # inside each directory from bathpath
					for subEntry in subSrcDirectories:
						#print(entry.name)
						if subEntry.is_file():
							#os.rename(basepath+'/'+entry.name,basepath+'/'+addToName+entry.name)
							#shutil.move(basepath+'/'+entry.name, basepath+'/'+dstDirectory)
							img= load_img(subEntry.path)
							img= img_to_array(img)
							#print(img.shape)
							sizes.append(img.shape)
							surfaces.append(img.shape[0]*img.shape[1])
			listOfSizes.append(sizes)
			listOfSurface.append(surfaces)
	#listOfSizes = np.array(listOfSizes)
	#listOfSurface = np.array(listOfSurface)
	return listOfSizes, listOfSurface



## Statistics anime dataset
## Calculate statistics apramenters: mean, median, std, Q1, Q3, some more
# input: ndarray
# output: mean, median, std, Q1, Q3, dif(median,Q1), dif(Q3,median), dif(median,256*256)
def calculates_statisticsParameters(inputArray):
	out_std = np.std(inputArray)
	out_mean = np.mean(inputArray)
	out_median = np.median(inputArray)
	out_Q1 = np.percentile(inputArray, 25)
	out_Q3 = np.percentile(inputArray, 75)
	out_dif_median_Q1 = abs(out_median - out_Q1)
	out_dif_Q3_meadian = abs(out_Q3 - out_median)
	out_dif_median_256x256 = abs(out_median - 256*256)
	print('median=%f\n' % out_median ,'mean=%f\n' % out_mean , 'std=%f\n' % out_std , 'Q1=%f\n' % out_Q1 , 'Q3=%f\n' % out_Q3, 
	'dif(median,Q1)=%f\t' % out_dif_median_Q1, 'dif(median,Q1)/median=%f%%\n' % ((out_dif_median_Q1/out_median)*100),
	'dif(Q3,median)=%f\t' % out_dif_Q3_meadian, 'dif(Q3,median)/median=%f%%\n' % ((out_dif_Q3_meadian/out_median)*100),
	'dif(median,256*256)=%f\t' % out_dif_median_256x256, 'dif(median,256*256)/median=%f%%' % ((out_dif_median_256x256/(out_median))*100))
	plt.boxplot(inputArray, vert=False)
	return out_mean, out_median, out_median, out_std, out_dif_median_Q1, out_dif_Q3_meadian, out_dif_median_256x256

realSelected_statistics = calculates_statisticsParameters(realSelected_surfaceList)

'''Characteristics:
model based image size = 256*256 = 65.536
statistics from surface (width*height)
>> Anime images statistics:
median=62068.000000
 mean=116161.502935
 std=154319.499516
 Q1=44274.000000
 Q3=136722.000000
 dif(median,Q1)=17794.000000	 dif(median,Q1)/median=28.668557%
 dif(Q3,median)=74654.000000	 dif(Q3,median)/median=120.277760%
 dif(median,256*256)=3468.000000	 dif(median,256*256)/median=5.587420%
> median =aprox 256*256
> top half images sufaces're much bigger than median
> indef of 256*256 = 273 => meddle 

>> Real images statistics:
median=184000.000000
 mean=181468.573567
 std=197557.244191
 Q1=165500.000000
 Q3=187500.000000
 dif(median,Q1)=18500.000000	 dif(median,Q1)/median=10.054348%
 dif(Q3,median)=3500.000000	 dif(Q3,median)/median=1.902174%
 dif(median,256*256)=118464.000000	 dif(median,256*256)/median=64.382609%
> close symmetrical (mean =aprox median)
> median is far from 256*256
> index of 256*256 = 1627 => at the beginning

>> Real images for each dog breed statistics
> index of 256*256 at the very beginning of each directory

Definitions to choose the real dog images to construct the final real image dataset:
As I only recolected 511 anime dog images, I must take only 500 samples from the real dogs dataset.
>> About breeds
> Size of dog similar to the anime's and to my dog (small one, 10kg)
> Hair of dog: short because of the anime's, balance between diversity and uniformity to include more posibles real dog photos when use the model
> breeds common on my residencial area
> Face features very remarkable, to help the model understand them
>> About images size
> I took images from original breed directories wich sizes aren't bigger than anime's Q3. 
> I replaced the images with human because the anime dataset doesn't have 
> 

>> Observations after run the code to select the real dog images to construct the final real image dataset:
> Because of the surface limits (<Q3) I didn't get to the 511 images. I completed manually with priority breeds
> Use the surface is a good aprox, but it would be better to use height and width limmits

>> Real dog selected imagess statistics
median=86436.000000
 mean=91754.418787
 std=30872.622574
 Q1=67500.000000
 Q2=115530.000000
 dif(median,Q1)=18936.000000	 dif(median,Q1)/median=21.907539%
 dif(Q3,median)=29094.000000	 dif(Q3,median)/median=33.659586%
 dif(median,256*256)=20900.000000	 dif(median,256*256)/median=24.179740%
> Q1>(256*256) (just a bit)
> 
'''


## To know ubication of 256*256 surface image in a sorted surface list
# To aply on interactive jupyter
min_dif_real = 256*256
index_min_real = 0
for i in range(len(real_surfaces)):
	dif = abs(real_surfaces[i]-256*256)
	if dif<min_dif_real:
		min_dif_real=dif
		index_min_real = i
		print(i, min_dif_real)

# Same but for each dog breed directories
for entry in surfaceList:
    print(entry[0])
    min_dif_real = 256*256
    index_min_real = 0
    aux = np.array(entry[1:])
    aux.sort(axis=-1)
    for i in range(len(aux)):
	    dif = abs(aux[i]-256*256)
	    if dif<min_dif_real:
		    min_dif_real=dif
		    index_min_real = i
    print(index_min_real, min_dif_real)

		
'''
            with os.scandir(entry) as images:   # inside basepath/iDirectory
                for file in images: # for each object in basepath/iDirectory
                    shutil.copy(file,dstDirectory)
    print('%s copied' % entry) 
'''



## plot one image
img = load_img('All_images/1fc355e99d4c3be91e5fb0c4a5ddfd2c.jpg', target_size=(256,256))
plt.imshow(img)
img = img_to_array(img)
print(img.shape)
plt.imshow(img)


## watching .mat files
test_data_mat = scipy.io.loadmat('lists/test_list.mat')
print(test_data_mat)

## load all images in a directory into memory
def load_images(path, size=(256,256)):
	data_list = list()
	# enumerate filenames in directory, assume all are images
	for filename in os.listdir(path):
		# load and resize the image
		pixels = load_img(path + filename, target_size=size)
		# convert to numpy array
		pixels = img_to_array(pixels)
		# store
		data_list.append(pixels)
	return np.asarray(data_list)

data_list = load_images('All_images/')



''' Code to move selected real dog images'''
## getting the criterions to select the breeds and imges from them
listOfWightedbreeds = list()
with open('Selected_criterion.csv', newline='') as csvfile:
	spamreader = csv.reader(csvfile, delimiter=',')
	for row in spamreader:
		listOfWightedbreeds.append([row[0],row[1]])

listOfWightedbreeds = np.array(listOfWightedbreeds)

## True if the directory is in the selected list
# input: directory name, ndarray selected list
# output: boolean
def is_DirectoryOnSelectedList(dirName, selectedList):
	if dirName in selectedList:
		return True
	else:
		return False

## True if image's size (surface) is in the range
# input: image path, range
# output: boolean
def is_imageSizeInRange(imgPath, range):
	img= load_img(imgPath)
	img= img_to_array(img)
	surface = img.shape[0]*img.shape[1]
	if (surface>range[0]) and (surface<range[1]):
		return True
	else:
		return False

## Return count of images to move from selected breeds
# input: directory name (index), list of selected breeds (directories)(ndarray)
# output: count of images (int)
def return_countImagesToMove(selectedDir, listDir):
	index = np.where(listDir == selectedDir)
	return listDir[index[0][0]][1]

## move selected real dog images to dst directory if satisfy selected breeds (directories) and size (surface range)
# input: src basepath, dst path, ndarray of selected breeds and how many images from each one, surface range
def move_selectedRealDogImages(basepath, dstDirectory, selectedList, range):
	with os.scandir(basepath) as srcDirectories: # inside basepath
		for entry in srcDirectories: # for each object in basepath
			if entry.is_dir() and is_DirectoryOnSelectedList(entry.name, selectedList):
				cantImgesToMove = int(return_countImagesToMove(entry.name, selectedList))
				print('Cant imgs to move: %d' % cantImgesToMove)
				i = 0
				with os.scandir(entry) as subSrcDirectories: # inside each directory from basepath
					for subEntry in subSrcDirectories:
						if i>cantImgesToMove:
							break
						if subEntry.is_file() and is_imageSizeInRange(subEntry.path, range) and i<cantImgesToMove:
							shutil.copy(subEntry.path, dstDirectory)
							print(subEntry.name)
							os.rename(dstDirectory+'/'+subEntry.name,dstDirectory+'/'+entry.name+'-'+subEntry.name)
							print(i)
							i=i+1
						

move_selectedRealDogImages('Images','All_images/real_selected',listOfWightedbreeds, [44274,136722])							

def count_iamgesPerBreed(selectedList, basepath):
	output_list = list()
	with os.scandir(basepath) as srcDirectories: # inside basepath
		for entry in srcDirectories: # for each object in basepath
			if entry.is_file():
				aux = str(entry.name)
				index = aux.find('-')
				aux = aux[0:index]
				# aux = np.array(aux)
				#print(aux,len(aux) , type(aux), output_list, type(output_list))
				if aux not in output_list:
					#np.append(output_list,[aux,1])
					output_list.append([aux,1])
				else:
					print('elif')
					#index_2 = np.where(output_list == aux)
					index_2 = output_list.index(aux)
					output_list[index_2,1] =+ 1 
				
	return output_list
				
coutOfImagesPerBreed = count_iamgesPerBreed(listOfWightedbreeds, 'All_images/real_selected')



# load a dataset as a list of two numpy arrays
## load images and create a ndarray: col_1->anime, col_2->real
# input: anime images path, real images path
# output: ndarray[countImages, 2]
def load_dataset (path, size=(256,256)):
  aux = list()
  with os.scandir(path) as SrcDirectories: # inside each directory from bathpath
    for Entry in SrcDirectories:
      if Entry.is_file():
        img= load_img(Entry.path, target_size=size)
        img= img_to_array(img)
        aux.append(img)

  return np.asarray(aux)

