from keras.models import load_model
import operator
import cv2
import sys, os

model=load_model("model1.h5")

cap = cv2.VideoCapture(0)


while True:
    _, frame = cap.read()
    # Simulating mirror image
    frame = cv2.flip(frame, 1)
    
    # Coordinates of the ROI
    x1 = int(0.5*frame.shape[1])
    y1 = 10
    x2 = frame.shape[1]-10
    y2 = int(0.5*frame.shape[1])
    # Drawing the ROI
    # The increment/decrement by 1 is to compensate for the bounding box
    cv2.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), (255,0,0) ,1)
    # Extracting the ROI
    roi = frame[y1:y2, x1:x2]
    
    # Resizing the ROI so it can be fed to the model for prediction
    roi = cv2.resize(roi, (28, 28)) 
    test_image = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    
    
    cv2.imshow("test",test_image)
    
    result = model.predict(test_image.reshape(1, 28,28, 1)/255)
    prediction = {'K':result[0][9],
                  'L':result[0][10],
                  'M':result[0][12],
                  'N':result[0][13],
                  'O':result[0][14],
                  'P':result[0][15],
                  'Q':result[0][16],
                  'R':result[0][17],
                  'S':result[0][18],
                  'T':result[0][11],
                  'U':result[0][19],
                  'V':result[0][20],
                  'W':result[0][21],
                  'X':result[0][22],
                  'Y':result[0][23]
                  }
        
    k = 0
    for i in range(65,74):
        prediction[chr(i)] = k
        k += 1

    prediction = dict()

    
    prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
    
    # Displaying the predictions
    cv2.putText(frame, prediction[0][0], (10, 120), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)    
    cv2.imshow("Frame", frame)
    
    interrupt = cv2.waitKey(10)
    if interrupt & 0xFF == 27: # esc key
        break
        
 
cap.release()
cv2.destroyAllWindows()