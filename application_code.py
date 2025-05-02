# # -*- coding: utf-8 -*-
# """
# Created on Thu Aug  8 00:13:32 2024

# @author: Air Gesture Team
# """

# #successfully implemented clicks Select action perform try very much close clicks performed

# ##All Three Implemented together successfully

# import cv2
# import mediapipe as mp
# from mediapipe.tasks import python
# from mediapipe.tasks.python import vision
# import os
# from mediapipe.framework.formats import landmark_pb2
# import time
# import pyautogui

# #for mouse horizontal scroll
# # import win32api,win32con


# baseOptions = mp.tasks.BaseOptions
# GestureRecognizer = mp.tasks.vision.GestureRecognizer
# GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
# GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
# VisionRunningMode = mp.tasks.vision.RunningMode

# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles

# # Variables to track gestures
# zoom_gesture_active = False
# initial_distance = None
# mouse_hover_active = False
# initial_position = None
# last_position_change_time = None

# mouse_scroll_active = False
# initial_middle_finger_tip_y = None
# initial_middle_finger_tip_x = None

# mouse_left_click_select_active = False
# initial_index_finger_tip_z = None
# gesture_active_time = None
# gesture_time = None


# def recognize_Action(gesture,hand_landmarks):
#     global zoom_gesture_active, initial_distance
#     global mouse_hover_active, initial_position, last_position_change_time
#     global mouse_scroll_active,initial_middle_finger_tip_y,initial_middle_finger_tip_x
#     global mouse_left_click_select_active,initial_index_finger_tip_z,gesture_time,gesture_active_time
    
#     thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
#     index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
#     middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
#     ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
#     pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

#     thumb_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
#     index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
#     middle_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
#     ring_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]
#     pinky_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]
    
#     if(gesture == "Open_Palm"):
#         print("Mouse move time")
#         if not mouse_hover_active:
#             mouse_hover_active = True
#             initial_position = (middle_tip.x,middle_tip.y,middle_tip.z)
#             print("Initial Position set to {}".format(initial_position))
#             last_position_change_time = time.time()
#             mouse_left_click_select_active =  False
#         return "move_mouse"
#     elif(gesture == "Closed_Fist"):
#         print("Scroll Gesture Activated.")
#         if not mouse_scroll_active:
#             mouse_scroll_active = True
#             initial_middle_finger_tip_y = middle_tip.y
#             initial_middle_finger_tip_x = middle_tip.x
#             print("Initial middle finger tip y-position set to {} x-position set to {}".format(initial_middle_finger_tip_y,initial_middle_finger_tip_x))
#             mouse_left_click_select_active = False
#             return "scroll_mouse"
#         elif mouse_scroll_active:
#             # Maintain the scroll action if it was previously activated
#             return "scroll_mouse"
#     # elif(gesture == "Pointing_Up"):
#     #     print("Click Gesture activated.")
#     #     if not mouse_left_click_select_active:
#     #         mouse_left_click_select_active = True
#     #         initial_index_finger_tip_z = index_tip.z
#     #         print("Initial Index finger tip z-position set to {}.".format(initial_index_finger_tip_z))
#     #         return "Left_select_click"
#     #     elif mouse_left_click_select_active:
#     #         print("current index z position. {}".format(index_tip.z))
#     #         return "Left_select_click"
    
#     elif(gesture == "Thumb_Up"):
#         print("Click Gesture activated.")
#         if not mouse_left_click_select_active:
#             mouse_left_click_select_active = True
#             gesture_time = time.time()
#             perform_click()
        
#     else:
#         mouse_hover_active = False
#         initial_position = None
#         last_position_change_time = None
        
#         mouse_scroll_active = False
#         initial_middle_finger_tip_y = None
#         initial_middle_finger_tip_x = None
        
#         mouse_left_click_select_active = False
#         initial_index_finger_tip_z = None
    
        
#     return None

# def perform_Action(action,hand_landmarks):
#     global initial_position, last_position_change_time, initial_distance
#     global initial_position_for_mouse_scroll
#     global mouse_left_click_select_active,initial_index_finger_tip_z
    
#     if(action == "move_mouse"):
#         if mouse_hover_active and initial_position:
#             current_position = (hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x,hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y,hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].z)
#             print("Current position = ",current_position)
#             move_mouse(current_position)
#     elif(action == "scroll_mouse"):
#         if mouse_scroll_active and (initial_middle_finger_tip_y or initial_middle_finger_tip_x) is not None:
#             current_middle_finger_tip_y = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
#             current_middle_finger_tip_x = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x
#             print("Current middle finger tip y-position {} x-position {}.".format(current_middle_finger_tip_y,current_middle_finger_tip_x))
#             perform_scroll(current_middle_finger_tip_y,current_middle_finger_tip_x)
#     elif(action == "Left_select_click"):
#         if mouse_left_click_select_active and initial_index_finger_tip_z is not None:
#             current_index_finger_tip_z = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].z
#             print("Current index finger tip z-position {}.".format(current_index_finger_tip_z))
#             perform_click(current_index_finger_tip_z)
            
            
# def perform_click():
#     global gesture_active_time,gesture_time
#     if not time.time() - gesture_time > 1:
#         pyautogui.click()
#         print("click performed.")
#         mouse_left_click_select_active = False
    
# # def perform_click(current_index_finger_tip_z):
# #     global initial_index_finger_tip_z,mouse_left_click_select_active
    
# #     print("current position {}, initial position {}".format(current_index_finger_tip_z,initial_index_finger_tip_z))
# #     z_movement = current_index_finger_tip_z - initial_index_finger_tip_z
    
# #     print(f"Z Movement: {z_movement}")
# #     print(f"abs Z Movement: {abs(z_movement)}")
    
# #     click_threshold = 0.005
    
# #     if abs(z_movement) > click_threshold:
# #         print("Click performed.")
# #         initial_index_finger_tip_z = current_index_finger_tip_z
# #         mouse_left_click_select_active = False
    
# def perform_scroll(current_middle_finger_tip_y,current_middle_finger_tip_x):
#     global initial_middle_finger_tip_y,initial_middle_finger_tip_x
    
#     print("current position {} {}, initial position {} {}".format(current_middle_finger_tip_y,current_middle_finger_tip_x,initial_middle_finger_tip_y,initial_middle_finger_tip_x))
    
#     y_movement = current_middle_finger_tip_y - initial_middle_finger_tip_y
#     x_movement = current_middle_finger_tip_x - initial_middle_finger_tip_x
#     print(f"Y Movement: {y_movement}")
#     print(f"X Movement: {x_movement}")
#     print(f"abs Y Movement: {abs(y_movement)}")
#     print(f"abs X Movement: {abs(x_movement)}")
    
#     scroll_threshold = 0.01
    
#     def scroll_horizontal(amount):
#         """
#         Perform horizontal scroll.
#         :param amount: Positive for right, negative for left.
#         """
#         win32api.mouse_event(win32con.MOUSEEVENTF_HWHEEL, 0, 0, amount, 0)
    
#     if current_middle_finger_tip_y < initial_middle_finger_tip_y and abs(y_movement) > scroll_threshold and abs(y_movement) > abs(x_movement):
#         pyautogui.scroll(-50)  # Scroll down`
#         print("Scrolling down")
#         initial_middle_finger_tip_y = current_middle_finger_tip_y
#         mouse_scroll_active = False
#     elif current_middle_finger_tip_y > initial_middle_finger_tip_y and abs(y_movement) > scroll_threshold and abs(y_movement) > abs(x_movement):
#         pyautogui.scroll(50)  # Scroll up
#         print("Scrolling up")
#         initial_middle_finger_tip_y = current_middle_finger_tip_y
#         mouse_scroll_active = False
#     elif current_middle_finger_tip_x < initial_middle_finger_tip_x and abs(x_movement) > abs(y_movement):
#         scroll_horizontal(-100)
#         print("Scrolling left")
#         initial_middle_finger_tip_x = current_middle_finger_tip_x
#         mouse_scroll_active = False
#     elif current_middle_finger_tip_x > initial_middle_finger_tip_x and abs(x_movement) > abs(y_movement):
#         scroll_horizontal(100)
#         print("Scrolling right")
#         initial_middle_finger_tip_x = current_middle_finger_tip_x
#         mouse_scroll_active = False
#     else:
#         print("Something is locha.")
    
        
# def move_mouse(current_position):
#     global initial_position, last_position_change_time
    
#     screen_width, screen_height = pyautogui.position()
#     print("Positions",current_position[0],current_position[1],current_position[2],initial_position[0],initial_position[1],initial_position[2])
    
#     delta_x = (current_position[0] - initial_position[0]) * screen_width 
#     delta_y = (current_position[1] - initial_position[1]) * screen_height 
    
#     print(f"Moving mouse by ({delta_x}, {delta_y})")
    
#     pyautogui.moveRel(delta_x,delta_y,duration=0.1)
    
#     initial_position = current_position
    
#     last_position_change_time = time.time()
    

# options = GestureRecognizerOptions(
#     base_options=baseOptions(model_asset_path="gesture_recognizer.task"),
#     running_mode=VisionRunningMode.IMAGE
# )

# with GestureRecognizer.create_from_options(options=options) as recognizer:
#     cap = cv2.VideoCapture(0)
    
#     while cap.isOpened():
#         success, raw_img = cap.read()
        
#         if not success:
#             break
        
#         RGB_image = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
        
#         result = recognizer.recognize(image=mp.Image(image_format=mp.ImageFormat.SRGB, data=RGB_image))
        
        
#         #print(result.gestures)      
#         #Gives formate output as gestures=[[Category(index=-1, score=0.7529258131980896, display_name='', category_name='Open_Palm')]]
        
#         gesture =None
        
#         try:
#             gesture = result.gestures[0][0].category_name
#             """
#             try:
#                 print(result.gestures[0][0].category_name)
#             except Exception:
#                 print("Gesture not Recognized successfully.")
#             """
#         except Exception:
#             print("No Hand detected",Exception)
            
#         annotated_image = raw_img.copy()
        
#         if result.hand_landmarks:
#             for hand_landmarks in result.hand_landmarks:
#                 hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
#                 hand_landmarks_proto.landmark.extend([
#                 landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
#               ])
#                 mp_drawing.draw_landmarks(annotated_image, hand_landmarks_proto,mp_hands.HAND_CONNECTIONS,mp_drawing_styles.get_default_hand_landmarks_style(),mp_drawing_styles.get_default_hand_connections_style())
                
#                 action = recognize_Action(gesture, hand_landmarks_proto)
                
#                 print("action = ",action)
                
#                 if(action):
#                     perform_Action(action, hand_landmarks_proto)
#         else:
#             # Reset gesture state when no hand is detected
#             zoom_gesture_active = False
#             initial_position = None
#             mouse_hover_active = False
#             initial_distance = None
#             last_position_change_time = None
            
#         # Check if mouse hover has been active and the position hasn't changed for 1 second
#         if mouse_hover_active and time.time() - last_position_change_time > 1:
#             mouse_hover_active = False
#             initial_position = None
            
        
        
#         cv2.putText(annotated_image, 'Detected Gesture : {}'.format(gesture), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            
            
        
#         cv2.imshow("Hand conection with landmarks", annotated_image) 
            
#         #GestureRecognizerResult(
#         #gestures=[[Category(index=-1, score=0.5390647649765015, display_name='', category_name='None')]], 
#         #handedness=[[Category(index=0, score=0.9785897135734558, display_name='Right', category_name='Right')]], 
#         #hand_landmarks=[[NormalizedLandmark(x=0.3165547847747803, y=0.8324751257896423, z=3.289906658210384e-08, visibility=0.0, presence=0.0), NormalizedLandmark(x=0.38971754908561707, y=0.8159221410751343, z=-0.006824594922363758, visibility=0.0, presence=0.0), NormalizedLandmark(x=0.436449259519577, y=0.7489752769470215, z=-0.006110101938247681, visibility=0.0, presence=0.0), NormalizedLandmark(x=0.4504600465297699, y=0.6829228401184082, z=-0.008332931436598301, visibility=0.0, presence=0.0), NormalizedLandmark(x=0.43288540840148926, y=0.6442659497261047, z=-0.010260134004056454, visibility=0.0, presence=0.0), NormalizedLandmark(x=0.4238666296005249, y=0.699932873249054, z=0.01814846880733967, visibility=0.0, presence=0.0), NormalizedLandmark(x=0.44068753719329834, y=0.6378264427185059, z=0.005792698357254267, visibility=0.0, presence=0.0), NormalizedLandmark(x=0.4427322447299957, y=0.6524844765663147, z=-0.009063294157385826, visibility=0.0, presence=0.0), NormalizedLandmark(x=0.43639102578163147, y=0.6839880347251892, z=-0.01749415509402752, visibility=0.0, presence=0.0), NormalizedLandmark(x=0.3966384828090668, y=0.6845641136169434, z=0.014832062646746635, visibility=0.0, presence=0.0), NormalizedLandmark(x=0.40909504890441895, y=0.6299698352813721, z=0.0031500370241701603, visibility=0.0, presence=0.0), NormalizedLandmark(x=0.41150814294815063, y=0.6531411409378052, z=-0.011263826861977577, visibility=0.0, presence=0.0), NormalizedLandmark(x=0.4014539420604706, y=0.6911287903785706, z=-0.019457632675766945, visibility=0.0, presence=0.0), NormalizedLandmark(x=0.36796584725379944, y=0.6733012795448303, z=0.008203726261854172, visibility=0.0, presence=0.0), NormalizedLandmark(x=0.3811940550804138, y=0.6232529878616333, z=-0.002726349513977766, visibility=0.0, presence=0.0), NormalizedLandmark(x=0.3861975073814392, y=0.6487790942192078, z=-0.011438911780714989, visibility=0.0, presence=0.0), NormalizedLandmark(x=0.3801654577255249, y=0.6873469352722168, z=-0.014805191196501255, visibility=0.0, presence=0.0), NormalizedLandmark(x=0.33741050958633423, y=0.6665089726448059, z=0.00029584349249489605, visibility=0.0, presence=0.0), NormalizedLandmark(x=0.34859371185302734, y=0.6239199042320251, z=-0.006093266420066357, visibility=0.0, presence=0.0), NormalizedLandmark(x=0.3535616993904114, y=0.6404908895492554, z=-0.007519966457039118, visibility=0.0, presence=0.0), NormalizedLandmark(x=0.3501628637313843, y=0.6721009612083435, z=-0.007065641228109598, visibility=0.0, presence=0.0)]], 
#         #hand_world_landmarks=[[Landmark(x=-0.038269784301519394, y=0.07385390251874924, z=-0.03829808533191681, visibility=0.0, presence=0.0), Landmark(x=-0.0003925058990716934, y=0.06468665599822998, z=-0.030460720881819725, visibility=0.0, presence=0.0), Landmark(x=0.027213623747229576, y=0.04214077815413475, z=-0.0296498853713274, visibility=0.0, presence=0.0), Landmark(x=0.043531328439712524, y=0.00866859219968319, z=-0.029088571667671204, visibility=0.0, presence=0.0), Landmark(x=0.03217123821377754, y=-0.01723933406174183, z=-0.02566632255911827, visibility=0.0, presence=0.0), Landmark(x=0.022802241146564484, y=0.007084866054356098, z=0.006639034021645784, visibility=0.0, presence=0.0), Landmark(x=0.034161437302827835, y=-0.010652841068804264, z=-0.0036467607133090496, visibility=0.0, presence=0.0), Landmark(x=0.037384968250989914, y=-0.006781895644962788, z=-0.014147446490824223, visibility=0.0, presence=0.0), Landmark(x=0.029640717431902885, y=0.009943543933331966, z=-0.021015150472521782, visibility=0.0, presence=0.0), Landmark(x=0.0012193935690447688, y=-0.0031171180307865143, z=0.010109120979905128, visibility=0.0, presence=0.0), Landmark(x=0.020756833255290985, y=-0.021621085703372955, z=-0.013924145139753819, visibility=0.0, presence=0.0), Landmark(x=0.019030459225177765, y=-0.00946531631052494, z=-0.02661304734647274, visibility=0.0, presence=0.0), Landmark(x=0.013513142243027687, y=0.009645032696425915, z=-0.030784688889980316, visibility=0.0, presence=0.0), Landmark(x=-0.014099261723458767, y=-0.006705684587359428, z=-0.005017818883061409, visibility=0.0, presence=0.0), Landmark(x=-0.00010548345744609833, y=-0.023448223248124123, z=-0.021668769419193268, visibility=0.0, presence=0.0), Landmark(x=0.00306902639567852, y=-0.011211292818188667, z=-0.03116229921579361, visibility=0.0, presence=0.0), Landmark(x=-0.0018313147593289614, y=0.010055416263639927, z=-0.034164246171712875, visibility=0.0, presence=0.0), Landmark(x=-0.03344298154115677, y=-0.00617549754679203, z=-0.01761523447930813, visibility=0.0, presence=0.0), Landmark(x=-0.023403221741318703, y=-0.02081146650016308, z=-0.028199972584843636, visibility=0.0, presence=0.0), Landmark(x=-0.01747501641511917, y=-0.017096513882279396, z=-0.03771096467971802, visibility=0.0, presence=0.0), Landmark(x=-0.018277013674378395, y=-0.0014074821956455708, z=-0.038303304463624954, visibility=0.0, presence=0.0)]])
        
        
        
        
#         if cv2.waitKey(5) & 0xFF == 27:
#             break
        
# cap.release()
# cv2.destroyAllWindows()                                                                                                                                                                                      

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# import cv2
# import mediapipe as mp
# from mediapipe.tasks import python
# from mediapipe.tasks.python import vision
# from mediapipe.framework.formats import landmark_pb2
# import time
# import pyautogui

# # Uncomment for Windows-only horizontal scroll
# # import win32api
# # import win32con

# # MediaPipe essentials
# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles

# # Gesture recognition config
# GestureRecognizer = mp.tasks.vision.GestureRecognizer
# GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
# VisionRunningMode = mp.tasks.vision.RunningMode
# BaseOptions = mp.tasks.BaseOptions

# # State variables
# mouse_hover_active = False
# mouse_scroll_active = False
# mouse_click_active = False
# initial_position = None
# initial_middle_y = None
# gesture_time = None

# # Move mouse based on middle finger tip movement
# def move_mouse(current_pos):
#     global initial_position
#     screen_w, screen_h = pyautogui.size()

#     dx = (current_pos[0] - initial_position[0]) * screen_w
#     dy = (current_pos[1] - initial_position[1]) * screen_h

#     pyautogui.moveRel(dx, dy, duration=0.05)
#     initial_position = current_pos

# # Vertical (and optional horizontal) scroll
# def perform_scroll(curr_y):
#     global initial_middle_y
#     y_movement = curr_y - initial_middle_y
#     scroll_threshold = 0.01

#     if abs(y_movement) > scroll_threshold:
#         if y_movement < 0:
#             pyautogui.scroll(-50)
#             print("Scrolling down")
#         else:
#             pyautogui.scroll(50)
#             print("Scrolling up")
#         initial_middle_y = curr_y

# # Perform click
# def perform_click():
#     global gesture_time, mouse_click_active
#     if time.time() - gesture_time < 1:
#         pyautogui.click()
#         print("Click performed")
#         mouse_click_active = False

# # Recognize action from gesture
# def recognize_action(gesture, landmarks):
#     global mouse_hover_active, mouse_scroll_active, mouse_click_active
#     global initial_position, initial_middle_y, gesture_time

#     middle_tip = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
#     index_tip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

#     if gesture == "Open_Palm":
#         if not mouse_hover_active:
#             mouse_hover_active = True
#             initial_position = (middle_tip.x, middle_tip.y)
#         return "move_mouse"

#     elif gesture == "Closed_Fist":
#         if not mouse_scroll_active:
#             mouse_scroll_active = True
#             initial_middle_y = middle_tip.y
#         return "scroll_mouse"

#     elif gesture == "Thumb_Up":
#         if not mouse_click_active:
#             mouse_click_active = True
#             gesture_time = time.time()
#             perform_click()

#     else:
#         # Reset states on unknown gesture
#         mouse_hover_active = False
#         mouse_scroll_active = False
#         mouse_click_active = False
#         initial_position = None
#         initial_middle_y = None

#     return None

# # Perform action based on recognized gesture
# def perform_action(action, landmarks):
#     middle_tip = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

#     if action == "move_mouse" and initial_position:
#         move_mouse((middle_tip.x, middle_tip.y))

#     elif action == "scroll_mouse":
#         perform_scroll(middle_tip.y)

# # Load MediaPipe Gesture Recognizer model
# options = GestureRecognizerOptions(
#     base_options=BaseOptions(model_asset_path="gesture_recognizer.task"),
#     running_mode=VisionRunningMode.IMAGE
# )

# # Run webcam + recognizer loop
# with GestureRecognizer.create_from_options(options) as recognizer:
#     cap = cv2.VideoCapture(0)

#     while cap.isOpened():
#         success, frame = cap.read()
#         if not success:
#             break

#         RGB_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         result = recognizer.recognize(
#             image=mp.Image(image_format=mp.ImageFormat.SRGB, data=RGB_image)
#         )

#         gesture = None
#         try:
#             gesture = result.gestures[0][0].category_name
#         except:
#             pass

#         annotated_frame = frame.copy()

#         if result.hand_landmarks:
#             for hand_landmarks in result.hand_landmarks:
#                 hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
#                 hand_landmarks_proto.landmark.extend([
#                     landmark_pb2.NormalizedLandmark(x=lmk.x, y=lmk.y, z=lmk.z)
#                     for lmk in hand_landmarks
#                 ])

#                 mp_drawing.draw_landmarks(
#                     annotated_frame,
#                     hand_landmarks_proto,
#                     mp_hands.HAND_CONNECTIONS,
#                     mp_drawing_styles.get_default_hand_landmarks_style(),
#                     mp_drawing_styles.get_default_hand_connections_style()
#                 )

#                 action = recognize_action(gesture, hand_landmarks_proto)
#                 if action:
#                     perform_action(action, hand_landmarks_proto)

#         cv2.putText(annotated_frame, f"Gesture: {gesture}", (10, 30),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

#         cv2.imshow("Gesture Control", annotated_frame)

#         if cv2.waitKey(5) & 0xFF == 27:
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
import time
import pyautogui

# MediaPipe essentials
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

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

# Move mouse based on middle finger tip movement (fixed mirror effect)
def move_mouse(current_pos):
    global initial_position
    screen_w, screen_h = pyautogui.size()

    # Convert MediaPipe coords to screen coords with mirror fix on x-axis
    adjusted_current_x = 1 - current_pos[0]
    adjusted_initial_x = 1 - initial_position[0]

    dx = (adjusted_current_x - adjusted_initial_x) * screen_w
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
            print("Scrolling down")
        else:
            pyautogui.scroll(50)
            print("Scrolling up")
        initial_middle_y = curr_y

def perform_click():
    global gesture_time, mouse_click_active
    if time.time() - gesture_time < 1:
        pyautogui.click()
        print("Click performed")
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

# Load Gesture Recognizer model
options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path="gesture_recognizer.task"),
    running_mode=VisionRunningMode.IMAGE
)

with GestureRecognizer.create_from_options(options) as recognizer:
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

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

        # Flip the display frame only for showing to user (safe here)
        display_frame = cv2.flip(annotated_frame, 1)

        cv2.putText(display_frame, f"Gesture: {gesture}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Gesture Control", display_frame)

        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
