 
import cv2

#hazir xml  nesne algilama dosyalari 

face_cascade = cv2.CascadeClassifier('haarcascade-frontalface-default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade-eye.xml')
eyeglasses_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')

####kotrollar her xml dosyasi icin eger yuklenmemis ise hata versin ve###
#ağız xml dosyasi icin kontrolu 
if eyeglasses_cascade.empty():
  raise IOError('Unable to load the glasses cascade classifier xml file')
#yüz xml dosyasi icin  kontrolu
if face_cascade.empty():
  raise IOError('Unable to load the face cascade classifier xml file')
#gözler xml dosyasi icin kontrolu 
if eye_cascade.empty():
  raise IOError('Unable to load the eye cascade classifier xml file')


videoCap= cv2.VideoCapture(0)

while True:
    _,img = videoCap.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    #yüz algilama
    faces = face_cascade.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w ,y+h),(255,0,0),2)
        roi_gray=gray[y:y+h, x:x+w]
        roi_color=img[y:y+h, x:x+w]
        
        #gözler algilama
        eyes=eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,9),2)
            
        #gözlük algilama    
        glasses = eyeglasses_cascade.detectMultiScale(roi_gray, 1.7, 11)
        for (mx,my,mw,mh) in glasses:
            cv2.rectangle(roi_color,(mx,my),(mx+mw,my+mh),(1,255,3),2)
                
            
            
            

    
    cv2.imshow('yuz algi',img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
videoCap.relase()
cv2.destroyAllWindows()
 























