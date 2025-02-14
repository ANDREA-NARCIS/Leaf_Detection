# implementation.py

import cv2
import numpy as np
from keras.models import load_model

# Load the trained model
model = load_model('leaf_model.h5')

# Create a list of class labels
class_labels = ['Alpinia Galanga (Rasna)', 'Amaranthus Viridis (Arive-Dantu)',
                'Artocarpus Heterophyllus (Jackfruit)', 'Azadirachta Indica (Neem)',
                'Basella Alba (Basale)', 'Brassica Juncea (Indian Mustard)',
                'Carissa Carandas (Karanda)', 'Citrus Limon (Lemon)',
                'Ficus Auriculata (Roxburgh fig)', 'Ficus Religiosa (Peepal Tree)',
                'Hibiscus Rosa-sinensis', 'Jasminum (Jasmine)',
                'Mangifera Indica (Mango)', 'Mentha (Mint)',
                'Moringa Oleifera (Drumstick)', 'Muntingia Calabura (Jamaica Cherry-Gasagase)',
                'Murraya Koenigii (Curry)', 'Nerium Oleander (Oleander)',
                'Nyctanthes Arbor-tristis (Parijata)', 'Ocimum Tenuiflorum (Tulsi)',
                'Piper Betle (Betel)', 'Plectranthus Amboinicus (Mexican Mint)',
                'Pongamia Pinnata (Indian Beech)', 'Psidium Guajava (Guava)',
                'Punica Granatum (Pomegranate)', 'Santalum Album (Sandalwood)',
                'Syzygium Cumini (Jamun)', 'Syzygium Jambos (Rose Apple)',
                'Tabernaemontana Divaricata (Crape Jasmine)', 'Trigonella Foenum-graecum (Fenugreek)']

# Initialize the flag for leaf detection
leaf_detected = False

# Start capturing video from the webcam
video_capture = cv2.VideoCapture(0)

while True:
    # Read the video frame
    ret, frame = video_capture.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to the grayscale image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Perform edge detection using Canny
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours of the edges
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Reset the leaf detection flag
    leaf_detected = False

    # Iterate over the contours
    for contour in contours:
        # Calculate the contour area
        area = cv2.contourArea(contour)

        # If the contour area is greater than a threshold, consider it as a leaf
        if area > 1000:
            # Get the bounding box coordinates of the contour
            x, y, w, h = cv2.boundingRect(contour)

            # Extract the region of interest (ROI) containing the leaf
            roi = frame[y:y + h, x:x + w]

            # Resize the ROI to match the input size of the model
            roi = cv2.resize(roi, (30, 30))

            # Preprocess the ROI
            processed_image = roi / 255.0
            processed_image = np.expand_dims(processed_image, axis=0)

            # Perform species prediction
            predictions = model.predict(processed_image)
            predicted_class_index = np.argmax(predictions)
            predicted_class = class_labels[predicted_class_index]
            prediction_confidence = predictions[0, predicted_class_index]

            # Draw a rectangle around the detected leaf
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Put the predicted species name and confidence on the frame
            text = f"{predicted_class} ({prediction_confidence:.2f})"
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Set the leaf detection flag to True
            leaf_detected = True

    # Show the video frame
    cv2.imshow('Leaf Detection', frame)

    # Wait for key press
    key = cv2.waitKey(1)

    # Capture the frame and perform leaf detection if 'S' is pressed
    if key == ord('S') or key == ord('s'):
        # Capture the frame
        captured_frame = frame.copy()

        # If a leaf is detected, perform leaf detection on the captured frame
        if leaf_detected:
            # Convert the captured frame to grayscale
            captured_gray = cv2.cvtColor(captured_frame, cv2.COLOR_BGR2GRAY)

            # Apply Gaussian blur to the captured grayscale image
            captured_blurred = cv2.GaussianBlur(captured_gray, (5, 5), 0)

            # Perform edge detection using Canny on the captured frame
            captured_edges = cv2.Canny(captured_blurred, 50, 150)

            # Find contours of the edges on the captured frame
            captured_contours, _ = cv2.findContours(captured_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Iterate over the captured contours
            for captured_contour in captured_contours:
                # Calculate the contour area
                captured_area = cv2.contourArea(captured_contour)

                # If the contour area is greater than a threshold, consider it as a leaf
                if captured_area > 1000:
                    # Get the bounding box coordinates of the contour
                    captured_x, captured_y, captured_w, captured_h = cv2.boundingRect(captured_contour)

                    # Extract the region of interest (ROI) containing the leaf from the captured frame
                    captured_roi = captured_frame[captured_y:captured_y + captured_h, captured_x:captured_x + captured_w]

                    # Resize the ROI to match the input size of the model
                    captured_roi = cv2.resize(captured_roi, (30, 30))

                    # Preprocess the ROI
                    captured_processed_image = captured_roi / 255.0
                    captured_processed_image = np.expand_dims(captured_processed_image, axis=0)

                    # Perform species prediction on the captured leaf
                    captured_predictions = model.predict(captured_processed_image)
                    captured_predicted_class_index = np.argmax(captured_predictions)
                    captured_predicted_class = class_labels[captured_predicted_class_index]
                    captured_prediction_confidence = captured_predictions[0, captured_predicted_class_index]

                    # Draw a rectangle around the detected leaf on the captured frame
                    cv2.rectangle(captured_frame, (captured_x, captured_y), (captured_x + captured_w, captured_y + captured_h),
                                  (0, 255, 0), 2)

                    # Put the predicted species name and confidence on the captured frame
                    text = f"{captured_predicted_class} ({captured_prediction_confidence:.2f})"
                    cv2.putText(captured_frame, text, (captured_x, captured_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                (0, 255, 0), 2)

            # Show the captured frame with leaf detection
            cv2.imshow('Captured Frame with Leaf Detection', captured_frame)

    # Exit the loop if 'Q' is pressed
    if key == ord('Q') or key == ord('q'):
        break

# Release the video capture and close all windows
video_capture.release()
cv2.destroyAllWindows()