import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import time

# MediaPipe essentials
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,  # Lower if latency is critical
    min_tracking_confidence=0.5    # Lower if latency is critical
)

# Gesture recognition function (example, can be extended)
def recognize_gesture(frame):
    # Flip the frame to match mirror image behavior
    frame = cv2.flip(frame, 1)
    RGB_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(RGB_image)
    gesture = None
    if result.multi_hand_landmarks:
        # Placeholder gesture detection based on landmarks (this can be customized)
        hand_landmarks = result.multi_hand_landmarks[0]
        # Example: Recognize a simple gesture based on thumb position
        thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
        index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        
        # If thumb and index fingers are close, it's a "thumb-index touch" gesture
        if abs(thumb_tip.x - index_tip.x) < 0.05 and abs(thumb_tip.y - index_tip.y) < 0.05:
            gesture = "Thumb-Index Touch"
        else:
            gesture = "Open Hand"
    
    return gesture, result

def run_gesture_control():
    cap = cv2.VideoCapture(0)  # Open webcam
    stframe = st.empty()  # Placeholder for frame display in Streamlit

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Recognize gesture from frame
        gesture, result = recognize_gesture(frame)

        annotated_frame = frame.copy()
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    annotated_frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

        # Display gesture feedback on screen
        cv2.putText(annotated_frame, f"Gesture: {gesture}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Show the annotated frame in Streamlit
        stframe.image(annotated_frame, channels="BGR")

    cap.release()  # Release the webcam when done

# Streamlit app UI
st.title("🖐️ Gesture Controlled Interface")

if st.button("▶️ Start Gesture Control"):
    run_gesture_control()

st.markdown("### Instructions:")
st.markdown("- **Thumb-Index Touch** → Detected when the thumb and index finger are close.")
st.markdown("- **Open Hand** → Detected when the hand is open.")
