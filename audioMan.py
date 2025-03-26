import warnings
warnings.filterwarnings("ignore")  # Suppress NumPy warnings

import cv2
import mediapipe as mp
import numpy as np
import sounddevice as sd
import librosa
import threading

# -- Hand Detection Setup --
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=2
)

# -- Load Audio --
audio_file = "Jupiter.wav"  # Replace with your .wav file
y, sr = librosa.load(audio_file, sr=None)

# -- Shared Variables --
speed_factor = 1.0
pitch_factor = 0.0
volume_factor = 1.0
lock = threading.Lock()

# -- Audio Chunking --
CHUNK_SIZE = 5000  # Larger chunk size to reduce choppiness
position = 0

def audio_callback(outdata, frames, time_info, status):
    global position
    with lock:
        sf = speed_factor
        pf = pitch_factor
        vol = volume_factor

    # Take a chunk from the original audio
    start = position
    end = start + CHUNK_SIZE
    chunk = y[start:end]

    # Loop back if we reached the end
    if len(chunk) < CHUNK_SIZE:
        position = 0
    else:
        position += CHUNK_SIZE

    # Speed & pitch on this chunk
    stretched = librosa.effects.time_stretch(chunk, rate=sf)
    pitched = librosa.effects.pitch_shift(stretched, sr=sr, n_steps=pf)

    # Apply volume
    pitched *= vol

    # Fill outdata
    out_len = len(pitched)
    if out_len < frames:
        outdata[:out_len] = pitched.reshape(-1, 1)
        outdata[out_len:] = 0
    else:
        outdata[:] = pitched[:frames].reshape(-1, 1)

# -- Start Output Stream --
stream = sd.OutputStream(
    samplerate=sr,
    blocksize=CHUNK_SIZE,
    channels=1,
    callback=audio_callback
)
stream.start()
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Track index fingertips for left and right
    left_index_pos = None
    right_index_pos = None

    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            hand_label = handedness.classification[0].label  # "Left" or "Right"

            # Index fingertip & thumb tip
            index_finger = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

            # Screen coordinates
            index_pos = (int(index_finger.x * frame.shape[1]),
                         int(index_finger.y * frame.shape[0]))
            thumb_pos = (int(thumb.x * frame.shape[1]),
                         int(thumb.y * frame.shape[0]))

            # Draw circles and line between index & thumb
            cv2.circle(frame, index_pos, 10, (255, 255, 255), -1)
            cv2.circle(frame, thumb_pos, 10, (255, 255, 255), -1)
            cv2.line(frame, index_pos, thumb_pos, (255, 255, 0), 2)

            if hand_label == "Left":
                left_index_pos = index_pos
            else:  # Right
                right_index_pos = index_pos

            # Distance for speed or pitch
            distance_index_thumb = np.linalg.norm(np.array(index_pos) - np.array(thumb_pos))
            # Update speed or pitch
            with lock:
                if hand_label == "Right":  
                    speed_factor = max(0.5, min(2.0, distance_index_thumb / 100))
                else:  
                    pitch_factor = max(-6, min(6, (distance_index_thumb - 50) / 10))

    # volume betweeen index
    if left_index_pos and right_index_pos:
        distance_lr = np.linalg.norm(np.array(left_index_pos) - np.array(right_index_pos))
        cv2.line(frame, left_index_pos, right_index_pos, (255, 255, 255), 2)

        with lock:
            volume_factor = max(0.0, min(2.0, distance_lr / 150.0))

    # Show Speed, Pitch, Volume onscreen
    cv2.putText(frame, f"Speed: {speed_factor:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(frame, f"Pitch: {pitch_factor:.2f}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(frame, f"Volume: {volume_factor:.2f}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Display Window
    cv2.imshow("Hand Tracking Audio Editor", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# -- Cleanup --
cap.release()
cv2.destroyAllWindows()
stream.stop()
stream.close()