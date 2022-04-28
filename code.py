from PIL import Image
import copy
import cv2
import numpy as np
from matplotlib import pyplot as plt
import os, os.path


"""
@calculateHistogram: This function calculates the histogram of a given image
@parameters: Image 
@returns: Histogram of the given image
"""
def calculateHistogram(image):

    height = image.shape[0]
    width = image.shape[1]

    histogram = np.zeros(256)

    for i in range(0, height):
        for j in range(0, width):
            intensity = image[i, j]
            histogram[intensity] += 1

    return histogram

"""
@histogramEqualization: This function applies the equalizing algorithm using the histogram of the image.
@parameters: Histogram and the Image 
@returns: Equalized image
"""
def histogramEqualization(histogram, image):
    height = image.shape[0]
    width = image.shape[1]

    cdf = np.zeros(256)
    #Normalize the cdf calculated
    cdf = np.cumsum(histogram)/(height*width)
    for i in range(0, height):
        for j in range(0, width):
            intensity = image[i, j]
            newIntensity = cdf[intensity]
            image[i, j] = int(newIntensity*255)

    return image
"""
@HistogramEqualizationOnly: This function applies histogram equalization to the given image
@parameters: List of images
@returns: No returns, saves the equalized images in designated folders.
"""
def HistogramEqualizationOnly(imgs):
    
 
    for i in range(len(imgs)):
        img = cv2.imread(imgs[i][0])
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        histogram = calculateHistogram(gray_img)
        img_result = histogramEqualization(histogram, gray_img)
        path = "Dataset_1/Histogram_Equalized"
        cv2.imwrite(os.path.join(path,imgs[i][1]),img_result)
"""
@AdpativeHistogramEqualization: This function applies adaptive histogram equalization to the given image
@parameters: List of images
@returns: No returns, saves the adaptive equalized images in designated folders.
"""
def AdpativeHistogramEqualization(imgs):

    for num in range(len(imgs)):
        img = cv2.imread(imgs[num][0])
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        x,y=img.shape
        #Dividing the image into 8x8 parts
        div1=round(x/8)
        div2=round(y/8)
        
        new_image = np.zeros_like(img)
        
        
        for i in range(0,8):
            
            for j in range(0,8):
                
                hist = calculateHistogram(img[div1*i:div1*(i+1),div2*j:div2*(j+1)])
                new_image[div1*i:div1*(i+1),div2*j:div2*(j+1)] = histogramEqualization(hist,img[div1*i:div1*(i+1),div2*j:div2*(j+1)])
                cv2.imshow("Result", new_image)
                cv2.waitKey(20)
        path = "Dataset_1/Adaptive_Histogram_Equalized"
        cv2.imwrite(os.path.join(path,imgs[num][1]),new_image)
"""
@CompareResults: This function calculates the Histograms of both the histogram equalized and adaptive histogram equalized images and compares them on a subplot.
@parameters: List of two sets of images
@returns: No returns, saves the results                                                                                                                                                     in designated folders.
"""
def CompareResults(imgs,list1,list2):

    for num in range(len(imgs)):


        original_img = cv2.imread(imgs[num][0])
        histogram_1 = calculateHistogram(original_img)
        
        Histogram_Equalized_Image = cv2.imread(list1[num][0])
        histogram_2 = calculateHistogram(Histogram_Equalized_Image)

        Adaptive_Histogram_Equalized_Image = cv2.imread(list2[num][0])
        histogram_3 = calculateHistogram(Adaptive_Histogram_Equalized_Image)


        fig = plt.figure(figsize=(20, 20))
        ax1 = fig.add_subplot(1,3,1)
        ax1.title.set_text('Histogram of Original Image')
        ax1.bar(np.arange(len(histogram_1)), histogram_1)

        ax2 = fig.add_subplot(1,3,2)
        ax2.title.set_text('Histogram of Histogram_Equalized Image')
        ax2.bar(np.arange(len(histogram_2)), histogram_2)

        ax3 = fig.add_subplot(1,3,3)
        ax3.title.set_text('Histogram of Adaptive_Histogram_Equalized Image')
        ax3.bar(np.arange(len(histogram_3)), histogram_3)
        path = "Dataset_1/Compare_Results"
        fig.savefig(os.path.join(path,imgs[num][1]))

   
        


if __name__ == "__main__":

    imgs = []
    #Defining the path
    path = "Dataset_1"
    #Checking for valid image types
    valid_images = [".jpeg",".gif",".png",".tga"]
    for f in os.listdir(path):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_images:
            continue
        #Appending the images into the list 
        imgs.append(((os.path.join(path,f)),f))
    
    HistogramEqualizationOnly(imgs)
    AdpativeHistogramEqualization(imgs)

    list1 = []
    path1 = "Dataset_1/Histogram_Equalized"
    valid_images = [".jpeg",".gif",".png",".tga"]
    for f in os.listdir(path):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_images:
            continue
        list1.append(((os.path.join(path1,f)),f))

    list2 = []
    #Defining the path
    path2 = "Dataset_1/Adaptive_Histogram_Equalized"
    valid_images = [".jpeg",".gif",".png",".tga"]
    for f in os.listdir(path):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_images:
            continue
        list2.append(((os.path.join(path2,f)),f))
    #Function Call for the comparision function
    CompareResults(imgs,list1,list2)
    
