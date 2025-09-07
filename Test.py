import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

# Initialize hand detector and classifier
detector = HandDetector(maxHands=1)
# Corrected the path to the model file (removed the trailing slash)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

# Parameters
offset = 20
imgSize = 300
labels = ["A", "B", "C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]# Make sure these labels match your labels.txt file

while True:
    success, img = cap.read()
    if not success:
        print("Error: Failed to capture image.")
        break
    
    # Create a copy of the image to draw on
    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Create a white background image
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        
        # Crop the hand image with a safe offset
        # Add checks to prevent cropping outside the frame
        y1 = max(0, y - offset)
        y2 = min(img.shape[0], y + h + offset)
        x1 = max(0, x - offset)
        x2 = min(img.shape[1], x + w + offset)
        
        imgCrop = img[y1:y2, x1:x2]

        # Check if imgCrop is valid before proceeding
        if imgCrop.shape[0] > 0 and imgCrop.shape[1] > 0:
            aspectRatio = h / w

            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize

            # --- CORRECTION: Perform prediction on the processed 'imgWhite' ---
            # Also moved this outside the if/else block so it runs for both cases
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            
            # --- IMPROVEMENT: Display the prediction on the screen ---
            cv2.rectangle(imgOutput, (x - offset, y - offset - 50), 
                          (x - offset + 90, y - offset), (255, 0, 255), cv2.FILLED)
            cv2.putText(imgOutput, labels[index], (x, y - 27), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
            cv2.rectangle(imgOutput, (x - offset, y - offset), 
                          (x + w + offset, y + h + offset), (255, 0, 255), 4)

            cv2.imshow("ImageCrop", imgCrop)
            cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", imgOutput)
    
    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()