import cv2
import numpy as np
import tensorflow as tf
print(tf.__version__)
from tensorflow.keras.models import load_model # type: ignore


model = load_model('model_path')


class_labels = {
    0: 'freshapples',
    1: 'freshbanana',
    2: 'freshorange',
    3: 'rottenapple',
    4: 'rottenbanana',
    5: 'rottenorange'
}


def preprocess_frame(frame):
    img_array = cv2.resize(frame, (224, 224))  
    img_array = img_array.astype('float32') / 255 
    img_array = np.expand_dims(img_array, axis=0)  
    return img_array


cap = cv2.VideoCapture(0)  

while True:
    ret, frame = cap.read()  
    if not ret:
        print("Failed to grab frame")
        break

    
    img = preprocess_frame(frame)

    
    prediction = model.predict(img)
    class_index = np.argmax(prediction)  
    predicted_class = class_labels[class_index]  

    
    cv2.putText(frame, f'Prediction: {predicted_class}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    
    cv2.imshow('Real-Time Freshness Detection', frame)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()