import  cv2

def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(r'D:\aconda\envs\tf_py2\Lib\site-packages\opencv\sources\data\haarcascades'
                                         r'\haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
    if len(faces) == 0:
        return None, None
    (x, y, w, h) = faces[0]

    img = cv2.rectangle(img, (x,y), (x+w,y+h) ,(0, 255, 0), 2)
    cv2.namedWindow("face detected")
    cv2.imshow("face detected", img)
    cv2.waitKey(0)

    return gray[y:y + w, x:x + h], faces[0]


test_img1 = cv2.imread('./index.jpg')
detect_face(test_img1)
