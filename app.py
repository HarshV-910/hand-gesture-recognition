# import streamlit as st
# import cv2
# import mediapipe as mp
# import numpy as np
# import time

# # MediaPipe essentials
# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles

# # Initialize hands
# hands = mp_hands.Hands(
#     static_image_mode=False,
#     max_num_hands=1,
#     min_detection_confidence=0.5,  # Lower if latency is critical
#     min_tracking_confidence=0.5    # Lower if latency is critical
# )

# # Gesture recognition function (example, can be extended)
# def recognize_gesture(frame):
#     # Flip the frame to match mirror image behavior
#     frame = cv2.flip(frame, 1)
#     RGB_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#     result = hands.process(RGB_image)
#     gesture = None
#     if result.multi_hand_landmarks:
#         # Placeholder gesture detection based on landmarks (this can be customized)
#         hand_landmarks = result.multi_hand_landmarks[0]
#         # Example: Recognize a simple gesture based on thumb position
#         thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
#         index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        
#         # If thumb and index fingers are close, it's a "thumb-index touch" gesture
#         if abs(thumb_tip.x - index_tip.x) < 0.05 and abs(thumb_tip.y - index_tip.y) < 0.05:
#             gesture = "Thumb-Index Touch"
#         else:
#             gesture = "Open Hand"
    
#     return gesture, result

# def run_gesture_control():
#     # Attempt to open webcam
#     cap = cv2.VideoCapture(0)  # Open webcam
#     if not cap.isOpened():
#         st.error("Webcam not accessible. Please ensure your webcam is connected and accessible.")
#         return

#     stframe = st.empty()  # Placeholder for frame display in Streamlit

#     while cap.isOpened():
#         success, frame = cap.read()
#         if not success:
#             break

#         # Recognize gesture from frame
#         gesture, result = recognize_gesture(frame)

#         annotated_frame = frame.copy()
#         if result.multi_hand_landmarks:
#             for hand_landmarks in result.multi_hand_landmarks:
#                 mp_drawing.draw_landmarks(
#                     annotated_frame,
#                     hand_landmarks,
#                     mp_hands.HAND_CONNECTIONS,
#                     mp_drawing_styles.get_default_hand_landmarks_style(),
#                     mp_drawing_styles.get_default_hand_connections_style()
#                 )

#         # Display gesture feedback on screen
#         cv2.putText(annotated_frame, f"Gesture: {gesture}", (10, 30),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

#         # Show the annotated frame in Streamlit
#         stframe.image(annotated_frame, channels="BGR")

#     cap.release()  # Release the webcam when done

# # Streamlit app UI
# st.title("🖐️ Gesture Controlled Interface")

# if st.button("▶️ Start Gesture Control"):
#     run_gesture_control()

# # Fallback for testing without webcam
# st.markdown("### Instructions:")
# st.markdown("- **Thumb-Index Touch** → Detected when the thumb and index finger are close.")
# st.markdown("- **Open Hand** → Detected when the hand is open.")
# st.markdown("\n\nIf you're unable to access the webcam, try testing the app locally with your webcam connected.")

import streamlit as st
import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

# MediaPipe essentials
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,  # Lower if latency is critical
    min_tracking_confidence=0.5 ,   # Lower if latency is critical
    model_complexity=0  # 0 is faster but less accurate, 1 is slower but more accurate
)


GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
BaseOptions = mp.tasks.BaseOptions

# State variables
mouse_hover_active = False
mouse_scroll_active = False
mouse_click_active = False
initial_position = None
initial_middle_y = None
gesture_time = None

# Move mouse based on middle finger tip movement
def move_mouse(current_pos):
    global initial_position
    screen_w, screen_h = pyautogui.size()

    # Convert MediaPipe coords to screen coords (no mirror fix needed)
    dx = (current_pos[0] - initial_position[0]) * screen_w
    dy = (current_pos[1] - initial_position[1]) * screen_h

    pyautogui.moveRel(dx, dy, duration=0.05)
    initial_position = (current_pos[0], current_pos[1])

# Vertical scroll
def perform_scroll(curr_y):
    global initial_middle_y
    y_movement = curr_y - initial_middle_y
    scroll_threshold = 0.01

    if abs(y_movement) > scroll_threshold:
        if y_movement < 0:
            pyautogui.scroll(-50)
        else:
            pyautogui.scroll(50)
        initial_middle_y = curr_y

def perform_click():
    global gesture_time, mouse_click_active
    if time.time() - gesture_time < 1:
        pyautogui.click()
        mouse_click_active = False

def recognize_action(gesture, landmarks):
    global mouse_hover_active, mouse_scroll_active, mouse_click_active
    global initial_position, initial_middle_y, gesture_time
    middle_tip = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    if gesture == "Open_Palm":
        if not mouse_hover_active:
            mouse_hover_active = True
            initial_position = (middle_tip.x, middle_tip.y)
        return "move_mouse"
    elif gesture == "Closed_Fist":
        if not mouse_scroll_active:
            mouse_scroll_active = True
            initial_middle_y = middle_tip.y
        return "scroll_mouse"
    elif gesture == "Thumb_Up":
        if not mouse_click_active:
            mouse_click_active = True
            gesture_time = time.time()
            perform_click()
    else:
        mouse_hover_active = False
        mouse_scroll_active = False
        mouse_click_active = False
        initial_position = None
        initial_middle_y = None
    return None

def perform_action(action, landmarks):
    middle_tip = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    if action == "move_mouse" and initial_position:
        move_mouse((middle_tip.x, middle_tip.y))
    elif action == "scroll_mouse":
        perform_scroll(middle_tip.y)

def run_gesture_control():
    options = GestureRecognizerOptions(
        base_options=BaseOptions(model_asset_path="gesture_recognizer.task"),
        running_mode=VisionRunningMode.IMAGE
    )

    with GestureRecognizer.create_from_options(options) as recognizer:
        cap = cv2.VideoCapture(0)

        stframe = st.empty()


        frame_skip = 2  # Process every 2nd frame
        frame_count = 0

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            if frame_count % frame_skip == 0:

    

                # Flip frame BEFORE recognition to match visual directions
                frame = cv2.flip(frame, 1)
                RGB_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                result = recognizer.recognize(
                    image=mp.Image(image_format=mp.ImageFormat.SRGB, data=RGB_image)
                )

                gesture = None
                try:
                    gesture = result.gestures[0][0].category_name
                except:
                    pass

                annotated_frame = frame.copy()

                if result.hand_landmarks:
                    for hand_landmarks in result.hand_landmarks:
                        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                        hand_landmarks_proto.landmark.extend([
                            landmark_pb2.NormalizedLandmark(x=lmk.x, y=lmk.y, z=lmk.z)
                            for lmk in hand_landmarks
                        ])
                        mp_drawing.draw_landmarks(
                            annotated_frame,
                            hand_landmarks_proto,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style()
                        )

                        action = recognize_action(gesture, hand_landmarks_proto)
                        if action:
                            perform_action(action, hand_landmarks_proto)
                
                cv2.putText(annotated_frame, f"Gesture: {gesture}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                stframe.image(annotated_frame, channels="BGR")

                if st.session_state.get("stop_gesture"):
                    break

            frame_count += 1
        cap.release()

# Streamlit app UI
st.title("🖐️ Gesture Controlled Mouse")

if "stop_gesture" not in st.session_state:
    st.session_state["stop_gesture"] = False

if st.button("▶️ Start Gesture Control"):
    st.session_state["stop_gesture"] = False
    run_gesture_control()

if st.button("🛑 Stop"):
    st.session_state["stop_gesture"] = True
    st.write("Stopped Gesture Control")

st.markdown("**Gesture Controls:**")
st.markdown("- 🖐️ **Open Palm** → Move Mouse")
st.markdown("- ✊ **Closed Fist** → Scroll")
st.markdown("- 👍 **Thumb Up** → Click")
