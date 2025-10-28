import cv2
import numpy as np
import mediapipe as mp
from math import hypot
import subprocess 
import time # New import for introducing a small delay

# --- Constants ---
MIN_DISTANCE = 30
MAX_DISTANCE = 200
MIN_CONTROL_VAL = 0
MAX_CONTROL_VAL = 100
# The standard brightness step size on macOS is about 6-7% per key press.
# We'll use this to calculate how many key presses are needed.
BRIGHTNESS_STEP_SIZE = 7 

# --- State Variable for Brightness ---
# We use a mutable list to track the last sent brightness level for comparison
current_brightness_level = [50] 

# --- Function Definitions ---

def set_mac_volume(level):
    """Sets the master volume level on macOS (0-100) using osascript."""
    try:
        subprocess.run(["osascript", "-e", f"set volume output volume {int(level)}"], 
                       check=True, 
                       capture_output=True)
    except subprocess.CalledProcessError as e:
        error_output = e.stderr.decode().strip()
        if "set volume output volume" not in error_output:
            print(f"macOS Volume control failed: {error_output}")
    except FileNotFoundError:
        print("Error: osascript not found.")


def set_mac_brightness_key_simulate(target_level, current_level_ref):
    """
    Simulates keyboard brightness up/down presses multiple times to achieve 
    a fast 0 to 100% change.
    """
    new_level = int(target_level)
    current_level = current_level_ref[0]
    
    # Calculate the total difference we need to cover
    level_difference = abs(new_level - current_level)
    
    # Determine how many key presses are needed (e.g., a 40% change needs 40/7 = 5.7, so 6 presses)
    num_presses = int(level_difference / BRIGHTNESS_STEP_SIZE)
    
    if num_presses > 0:
        key_code = ""
        
        if new_level > current_level:
            # F15 is often the Brightness UP key code on macOS
            key_code = "113" 
        elif new_level < current_level:
            # F14 is often the Brightness DOWN key code on macOS
            key_code = "107"
            
        if key_code:
            try:
                # Send the key press command multiple times
                for _ in range(num_presses):
                    subprocess.run(["osascript", "-e", f"tell application \"System Events\" to key code {key_code}"],
                                   check=False, # Set to False as osascript might fail when hitting boundaries
                                   capture_output=True)
                    # Introduce a very small delay (optional, but helps system register commands)
                    time.sleep(0.01) 
                
                # Update the reference level to the new target level
                current_level_ref[0] = new_level
            except Exception as e:
                print(f"Brightness key press failed: {e}")


# --- Initialization ---

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# --- Main Loop ---

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    success, img = cap.read()
    if not success:
        print("Error: Could not read frame.")
        break
        
    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks and results.multi_handedness:
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            hand_label = results.multi_handedness[i].classification[0].label
            
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get coordinates for Thumb Tip and Index Finger Tip
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            h, w, _ = img.shape
            thumb_pos = (int(thumb_tip.x * w), int(thumb_tip.y * h))
            index_pos = (int(index_tip.x * w), int(index_tip.y * h))

            # Draw visuals
            cv2.circle(img, thumb_pos, 10, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, index_pos, 10, (255, 0, 0), cv2.FILLED)
            cv2.line(img, thumb_pos, index_pos, (0, 255, 0), 3)

            # Calculate the distance
            distance = hypot(index_pos[0] - thumb_pos[0], index_pos[1] - thumb_pos[1])

            # Volume Control with Right Hand
            if hand_label == "Right":
                vol_percent = np.interp(distance, [MIN_DISTANCE, MAX_DISTANCE], [MIN_CONTROL_VAL, MAX_CONTROL_VAL])
                
                set_mac_volume(vol_percent)

                # Volume Bar Visuals
                vol_bar = np.interp(distance, [MIN_DISTANCE, MAX_DISTANCE], [400, 150])
                cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 2)
                cv2.rectangle(img, (50, int(vol_bar)), (85, 400), (255, 0, 0), cv2.FILLED)
                cv2.putText(img, f'Vol: {int(vol_percent)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)

            # Brightness Control with Left Hand (NEW: Multiple Key Presses)
            elif hand_label == "Left":
                brightness = np.interp(distance, [MIN_DISTANCE, MAX_DISTANCE], [MIN_CONTROL_VAL, MAX_CONTROL_VAL])
                
                # Call the function that sends multiple key presses
                set_mac_brightness_key_simulate(brightness, current_brightness_level)

                # Brightness Bar Visuals
                brightness_bar = np.interp(current_brightness_level[0], [MIN_CONTROL_VAL, MAX_CONTROL_VAL], [400, 150])
                cv2.rectangle(img, (100, 150), (135, 400), (0, 255, 0), 2)
                cv2.rectangle(img, (100, int(brightness_bar)), (135, 400), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, f'Brightness: {int(current_brightness_level[0])} %', (90, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 3)
                
    cv2.imshow("Gesture Controller (macOS Native)", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()