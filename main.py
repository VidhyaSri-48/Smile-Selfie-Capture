import cv2  #importing opencv package to work with video analysis and image processing

video = cv2.VideoCapture(0) #video capture function is used to open the real time camera connected to the system, (0) is used to detect the camera which is connected, can also give the path of the video in that '0' place
faceCascade = cv2.CascadeClassifier("dataset/haarcascade_frontalface_default.xml")#pretrained model is imported to detect the face which is being captured by the camera,an external file is imported(downloaded from github).
smileCascade = cv2.CascadeClassifier("dataset/haarcascade_smile.xml")#pretrained model is imported to detect the smile which is being captured by the camera and detected face,an external file is imported(downloaded from github).

while True:# while the camera is working
    success,img = video.read()#read the capturing video 
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#convert the capturing video into gray scle format
    faces = faceCascade.detectMultiScale(grayImg,1.1,4)#detecting the face using the imported pre-trained model
    cnt=1#initially setting the count as 1 (the captured image will be saved in the name of the count
    keyPressed = cv2.waitKey(1)#wait key function is used to close the camera when the defined key is pressed
    for x,y,w,h in faces: # for the detected face
        #img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,0),3) #drawing the rectangle around the face using rectangle function in opencv-python
        smiles = smileCascade.detectMultiScale(grayImg,1.8,15)#detecting the smile using imported haarcascade smile detection pretrained model
        
        for x,y,w,h in smiles:#for the detected smile
            #img = cv2.rectangle(img,(x,y),(x+w,y+h),(100,100,100),5) #drawing the rectangle around the face using rectangle function in opencv-python
            print("Image "+str(cnt)+"Saved") #printing the saved images names
            path=r'D:\SUBHASHRI\internship\subhashri in sprout\subhashri in sprout\project\smile-selfie-capture-project\dataset'+str(cnt)+'.png' #defining a path save the captured images with smile
            cv2.imwrite(path,img)#saving the captured image
            cnt +=1#after saving one image adding the count to save the next image with different names
            if(cnt>=503):   #if the count is greater than or equal to 503 then stop capturing
                break
                
    cv2.imshow('live video',img)#title for the camera opencv window
    if(keyPressed & 0xFF==ord('a')):# if the key a is pressed then the camera will be closed
        break

video.release()   #to ensure that the camera is properly released to free up the system's space                               
cv2.destroyAllWindows() #closes all the open cv windows

'''cv2.rectangle(image, start_point, end_point, color, thickness)
 - image: the image on which the rectangle will be drawn
 - start_point: the starting point (x, y) coordinates of the rectangle
 - end_point: the ending point (x, y) coordinates of the rectangle
 - color: the color of the rectangle in BGR format
 - thickness: the thickness of the rectangle outlined. If you want to fill the rectangle, use the value cv2.FILLED.'''
