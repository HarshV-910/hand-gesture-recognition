import streamlit as st
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
import time
import pyautogui
import numpy as np
from threading import Thread
import queue

# Page configuration
st.set_page_config(
    page_title="Gesture Recognition Control",
    page_icon="👋",
    layout="wide"
)

# MediaPipe essentials
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
BaseOptions = mp.tasks.BaseOptions

class GestureController:
    def __init__(self):
        self.mouse_hover_active = False
        self.mouse_scroll_active = False
        self.mouse_click_active = False
        self.initial_position = None
        self.initial_middle_y = None
        self.gesture_time = None
        self.is_running = False
        self.cap = None
        self.recognizer = None
        
    def setup_recognizer(self):
        """Initialize the gesture recognizer"""
        try:
            options = GestureRecognizerOptions(
                base_options=BaseOptions(model_asset_path="gesture_recognizer.task"),
                running_mode=VisionRunningMode.IMAGE
            )
            self.recognizer = GestureRecognizer.create_from_options(options)
            return True
        except Exception as e:
            st.error(f"Error loading gesture recognizer model: {e}")
            st.error("Please ensure 'gesture_recognizer.task' file is in the same directory")
            return False

    def move_mouse(self, current_pos):
        """Move mouse based on middle finger tip movement"""
        screen_w, screen_h = pyautogui.size()
        
        # Convert MediaPipe coords to screen coords with mirror fix on x-axis
        adjusted_current_x = 1 - current_pos[0]
        adjusted_initial_x = 1 - self.initial_position[0]
        
        dx = (adjusted_current_x - adjusted_initial_x) * screen_w
        dy = (current_pos[1] - self.initial_position[1]) * screen_h
        
        pyautogui.moveRel(dx, dy, duration=0.05)
        self.initial_position = (current_pos[0], current_pos[1])

    def perform_scroll(self, curr_y):
        """Perform vertical scroll"""
        y_movement = curr_y - self.initial_middle_y
        scroll_threshold = 0.01
        
        if abs(y_movement) > scroll_threshold:
            if y_movement < 0:
                pyautogui.scroll(-3)  # Reduced scroll speed
            else:
                pyautogui.scroll(3)   # Reduced scroll speed
            self.initial_middle_y = curr_y

    def perform_click(self):
        """Perform mouse click"""
        if time.time() - self.gesture_time < 1:
            pyautogui.click()
            self.mouse_click_active = False

    def recognize_action(self, gesture, landmarks):
        """Recognize and set action based on gesture"""
        middle_tip = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        
        if gesture == "Open_Palm":
            if not self.mouse_hover_active:
                self.mouse_hover_active = True
                self.initial_position = (middle_tip.x, middle_tip.y)
            return "move_mouse"
        
        elif gesture == "Closed_Fist":
            if not self.mouse_scroll_active:
                self.mouse_scroll_active = True
                self.initial_middle_y = middle_tip.y
            return "scroll_mouse"
        
        elif gesture == "Thumb_Up":
            if not self.mouse_click_active:
                self.mouse_click_active = True
                self.gesture_time = time.time()
                self.perform_click()
        
        else:
            self.mouse_hover_active = False
            self.mouse_scroll_active = False
            self.mouse_click_active = False
            self.initial_position = None
            self.initial_middle_y = None
        
        return None

    def perform_action(self, action, landmarks):
        """Perform the recognized action"""
        middle_tip = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        
        if action == "move_mouse" and self.initial_position:
            self.move_mouse((middle_tip.x, middle_tip.y))
        elif action == "scroll_mouse":
            self.perform_scroll(middle_tip.y)

    def start_camera(self):
        """Start camera capture"""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            st.error("Error: Could not open camera")
            return False
        return True

    def stop_camera(self):
        """Stop camera capture"""
        self.is_running = False
        if self.cap:
            self.cap.release()
        if self.recognizer:
            self.recognizer.close()

    def process_frame(self):
        """Process a single frame"""
        if not self.cap or not self.recognizer:
            return None, None
            
        success, frame = self.cap.read()
        if not success:
            return None, None
            
        RGB_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        result = self.recognizer.recognize(
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
                
                action = self.recognize_action(gesture, hand_landmarks_proto)
                if action:
                    self.perform_action(action, hand_landmarks_proto)
        
        # Flip the display frame for user
        display_frame = cv2.flip(annotated_frame, 1)
        
        # Add gesture text
        cv2.putText(display_frame, f"Gesture: {gesture}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return display_frame, gesture

# Initialize session state
if 'controller' not in st.session_state:
    st.session_state.controller = GestureController()
if 'camera_running' not in st.session_state:
    st.session_state.camera_running = False

# Main UI
st.title("👋 Gesture Recognition Control")
st.markdown("Control your mouse using hand gestures!")

# Main layout
left_col, right_col = st.columns([1, 2])

with left_col:
    # Start button
    if st.button("🎥 Start Camera", disabled=st.session_state.camera_running, use_container_width=True):
        if st.session_state.controller.setup_recognizer():
            if st.session_state.controller.start_camera():
                st.session_state.camera_running = True
                st.session_state.controller.is_running = True
                st.rerun()
    
    # Stop button
    if st.button("⏹️ Stop Camera", disabled=not st.session_state.camera_running, use_container_width=True):
        st.session_state.controller.stop_camera()
        st.session_state.camera_running = False
        st.rerun()

    # Instructions at the bottom
    st.markdown("""
    ---
    **Supported Gestures:**
    * 🖐️ **Open Palm**: Move mouse cursor
    * ✊ **Closed Fist**: Scroll up/down
    * 👍 **Thumb Up**: Click""")
    
with right_col:
    # Camera feed display
    if st.session_state.camera_running:
        # Create placeholder for video stream
        frame_placeholder = st.empty()
        
        # Process frames in a loop
        while st.session_state.camera_running:
            frame, gesture = st.session_state.controller.process_frame()
            
            if frame is not None:
                # Convert BGR to RGB for streamlit
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame_rgb, channels="RGB")
            
            # Small delay to prevent excessive CPU usage
            time.sleep(0.033)  # ~30 FPS
    else:
        st.info("Click 'Start Camera' to begin gesture recognition")


# Cleanup on app close
if st.session_state.camera_running:
    # Register cleanup function
    import atexit
    atexit.register(st.session_state.controller.stop_camera)