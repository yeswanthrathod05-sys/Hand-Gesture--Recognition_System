import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np
import pickle
import mediapipe as mp
from PIL import Image, ImageTk


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "models/gesture_model.pkl"
LANDMARK_MODEL_PATH = "hand_landmarker.task"

GESTURES = [
    ("✊", "FIST"),
    ("✋", "PALM"),
    ("👍", "THUMBS UP"),
    ("✌", "VICTORY"),
    ("☝", "ONE FINGER")
]


# ============================================================
# LOAD MODEL
# ============================================================

with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)


# ============================================================
# MEDIAPIPE HAND LANDMARKER
# ============================================================

BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=LANDMARK_MODEL_PATH
    ),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

landmarker = mp.tasks.vision.HandLandmarker.create_from_options(
    options
)


# ============================================================
# MAIN WINDOW
# ============================================================

root = tk.Tk()

root.title("Hand Gesture Recognition System")
root.geometry("1100x800")
root.configure(bg="#1e1e1e")
root.resizable(False, False)


# ============================================================
# COLORS
# ============================================================

BG = "#1e1e1e"
HEADER = "#252526"
PANEL = "#252526"
CARD = "#2d2d30"
TEXT = "#ffffff"
SECONDARY = "#bdbdbd"
GREEN = "#4CAF50"
RED = "#f44336"


# ============================================================
# VARIABLES
# ============================================================

cap = None
camera_running = False

gesture_var = tk.StringVar(
    value="NO HAND DETECTED"
)

confidence_var = tk.StringVar(
    value="Confidence: 0%"
)

status_var = tk.StringVar(
    value="System Ready"
)

prediction_history = []


# ============================================================
# HEADER
# ============================================================

header = tk.Frame(
    root,
    bg=HEADER,
    height=85
)

header.pack(fill="x")

title_label = tk.Label(
    header,
    text="HAND GESTURE RECOGNITION SYSTEM",
    font=("Arial", 22, "bold"),
    fg=TEXT,
    bg=HEADER
)

title_label.pack(
    pady=(15, 2)
)

subtitle_label = tk.Label(
    header,
    text="Real-Time Computer Vision System",
    font=("Arial", 11),
    fg=SECONDARY,
    bg=HEADER
)

subtitle_label.pack()


# ============================================================
# MAIN FRAME
# ============================================================

main_frame = tk.Frame(
    root,
    bg=BG
)

main_frame.pack(
    fill="both",
    expand=True,
    padx=20,
    pady=15
)


# ============================================================
# CAMERA PANEL
# ============================================================

camera_panel = tk.Frame(
    main_frame,
    bg=PANEL,
    width=680,
    height=500
)

camera_panel.pack(
    side="left",
    fill="y",
    padx=(0, 15)
)

camera_panel.pack_propagate(False)

camera_title = tk.Label(
    camera_panel,
    text="LIVE CAMERA",
    font=("Arial", 14, "bold"),
    fg=TEXT,
    bg=PANEL
)

camera_title.pack(
    anchor="w",
    padx=15,
    pady=(15, 10)
)

camera_label = tk.Label(
    camera_panel,
    text="Camera is stopped",
    font=("Arial", 14),
    fg=SECONDARY,
    bg="#111111"
)

camera_label.pack(
    padx=15,
    pady=5,
    fill="both",
    expand=True
)


# ============================================================
# DETECTION PANEL
# ============================================================

detection_panel = tk.Frame(
    main_frame,
    bg=PANEL,
    width=350,
    height=500
)

detection_panel.pack(
    side="right",
    fill="y"
)

detection_panel.pack_propagate(False)


detection_title = tk.Label(
    detection_panel,
    text="DETECTION",
    font=("Arial", 14, "bold"),
    fg=TEXT,
    bg=PANEL
)

detection_title.pack(
    anchor="w",
    padx=15,
    pady=(15, 15)
)


# ============================================================
# DETECTED GESTURE BOX
# ============================================================

result_box = tk.Frame(
    detection_panel,
    bg="#202020"
)

result_box.pack(
    fill="x",
    padx=15,
    pady=(0, 12)
)


result_heading = tk.Label(
    result_box,
    text="DETECTED GESTURE",
    font=("Arial", 9, "bold"),
    fg=SECONDARY,
    bg="#202020"
)

result_heading.pack(
    pady=(10, 2)
)


gesture_label = tk.Label(
    result_box,
    textvariable=gesture_var,
    font=("Arial", 19, "bold"),
    fg=GREEN,
    bg="#202020"
)

gesture_label.pack(
    pady=(2, 10)
)


# ============================================================
# CONFIDENCE
# ============================================================

confidence_label = tk.Label(
    detection_panel,
    textvariable=confidence_var,
    font=("Arial", 11, "bold"),
    fg=TEXT,
    bg=PANEL
)

confidence_label.pack(
    anchor="w",
    padx=15
)


style = ttk.Style()

style.theme_use("default")

style.configure(
    "Confidence.Horizontal.TProgressbar",
    troughcolor="#3a3a3a",
    background=GREEN,
    bordercolor="#3a3a3a",
    lightcolor=GREEN,
    darkcolor=GREEN,
    thickness=11
)


confidence_bar = ttk.Progressbar(
    detection_panel,
    style="Confidence.Horizontal.TProgressbar",
    orient="horizontal",
    length=315,
    mode="determinate",
    maximum=100,
    value=0
)

confidence_bar.pack(
    padx=15,
    pady=(6, 12)
)


# ============================================================
# SUPPORTED GESTURES
# ============================================================

supported_title = tk.Label(
    detection_panel,
    text="SUPPORTED GESTURES",
    font=("Arial", 9, "bold"),
    fg=SECONDARY,
    bg=PANEL
)

supported_title.pack(
    anchor="w",
    padx=15,
    pady=(0, 5)
)


cards_frame = tk.Frame(
    detection_panel,
    bg=PANEL
)

cards_frame.pack(
    fill="x",
    padx=15
)


gesture_cards = {}


# ============================================================
# HIGHLIGHT FUNCTION
# ============================================================

def highlight_gesture(gesture):

    for name, card in gesture_cards.items():

        if name == gesture:

            card.config(bg=GREEN)

            for widget in card.winfo_children():
                widget.config(bg=GREEN)

        else:

            card.config(bg=CARD)

            for widget in card.winfo_children():
                widget.config(bg=CARD)


# ============================================================
# CREATE GESTURE CARDS
# ============================================================

for emoji, name in GESTURES:

    card = tk.Frame(
        cards_frame,
        bg=CARD,
        height=38
    )

    card.pack(
        fill="x",
        pady=2
    )

    card.pack_propagate(False)


    icon = tk.Label(
        card,
        text=emoji,
        font=("Arial", 16),
        fg=TEXT,
        bg=CARD,
        width=4
    )

    icon.pack(
        side="left",
        padx=(5, 0)
    )


    name_label = tk.Label(
        card,
        text=name,
        font=("Arial", 10, "bold"),
        fg=TEXT,
        bg=CARD
    )

    name_label.pack(
        side="left",
        padx=8
    )


    gesture_cards[name] = card


# ============================================================
# PREDICTION HISTORY
# ============================================================

history_title = tk.Label(
    detection_panel,
    text="LAST 5 PREDICTIONS",
    font=("Arial", 9, "bold"),
    fg=SECONDARY,
    bg=PANEL
)

history_title.pack(
    anchor="w",
    padx=15,
    pady=(7, 4)
)


history_frame = tk.Frame(
    detection_panel,
    bg="#202020"
)

history_frame.pack(
    fill="x",
    padx=15
)


history_labels = []


for i in range(5):

    label = tk.Label(
        history_frame,
        text="—",
        font=("Arial", 9),
        fg=SECONDARY,
        bg="#202020",
        anchor="w"
    )

    label.pack(
        fill="x",
        padx=8,
        pady=1
    )

    history_labels.append(label)


# ============================================================
# ADD HISTORY
# ============================================================

def add_prediction_history(
    gesture,
    confidence
):

    entry = (
        f"{gesture}    {confidence:.1f}%"
    )

    prediction_history.insert(
        0,
        entry
    )

    if len(prediction_history) > 5:

        prediction_history.pop()


    for i in range(5):

        if i < len(prediction_history):

            history_labels[i].config(
                text=prediction_history[i],
                fg=TEXT
            )

        else:

            history_labels[i].config(
                text="—",
                fg=SECONDARY
            )


# ============================================================
# CLEAR HISTORY
# ============================================================

def clear_history():

    prediction_history.clear()

    for label in history_labels:

        label.config(
            text="—",
            fg=SECONDARY
        )


# ============================================================
# CLEAR HISTORY BUTTON
# ============================================================

clear_history_button = tk.Button(
    detection_panel,
    text="CLEAR HISTORY",
    command=clear_history,
    font=("Arial", 8, "bold"),
    bg="#444444",
    fg="white",
    activebackground="#555555",
    activeforeground="white",
    relief="flat",
    cursor="hand2"
)

clear_history_button.pack(
    fill="x",
    padx=15,
    pady=(5, 0)
)


# ============================================================
# SYSTEM STATUS
# ============================================================

status_frame = tk.Frame(
    detection_panel,
    bg="#202020"
)

status_frame.pack(
    fill="x",
    padx=15,
    pady=(7, 0)
)


status_heading = tk.Label(
    status_frame,
    text="SYSTEM STATUS",
    font=("Arial", 8, "bold"),
    fg=SECONDARY,
    bg="#202020"
)

status_heading.pack(
    anchor="w",
    padx=10,
    pady=(4, 1)
)


status_label = tk.Label(
    status_frame,
    textvariable=status_var,
    font=("Arial", 9, "bold"),
    fg=GREEN,
    bg="#202020"
)

status_label.pack(
    anchor="w",
    padx=10,
    pady=(0, 4)
)


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(hand_landmarks):

    landmarks = []

    for landmark in hand_landmarks:

        landmarks.append([
            landmark.x,
            landmark.y,
            landmark.z
        ])


    landmarks = np.array(
        landmarks
    )

    wrist = landmarks[0].copy()

    landmarks = landmarks - wrist

    max_value = np.max(
        np.abs(landmarks)
    )

    if max_value > 0:

        landmarks = (
            landmarks / max_value
        )


    return landmarks.flatten().reshape(
        1,
        -1
    )


# ============================================================
# START CAMERA
# ============================================================

def start_camera():

    global cap
    global camera_running

    if camera_running:
        return


    cap = cv2.VideoCapture(0)


    if not cap.isOpened():

        status_var.set(
            "Camera Error"
        )

        status_label.config(
            fg=RED
        )

        gesture_var.set(
            "CAMERA ERROR"
        )

        return


    camera_running = True

    status_var.set(
        "Camera Running"
    )

    status_label.config(
        fg=GREEN
    )

    update_camera()


# ============================================================
# STOP CAMERA
# ============================================================

def stop_camera():

    global cap
    global camera_running

    camera_running = False


    if cap is not None:

        cap.release()
        cap = None


    camera_label.config(
        image="",
        text="Camera is stopped"
    )


    gesture_var.set(
        "NO HAND DETECTED"
    )

    confidence_var.set(
        "Confidence: 0%"
    )

    confidence_bar["value"] = 0


    highlight_gesture("")


    status_var.set(
        "System Stopped"
    )

    status_label.config(
        fg=RED
    )


# ============================================================
# UPDATE CAMERA
# ============================================================

def update_camera():

    global cap
    global camera_running

    if not camera_running:
        return

    if cap is None:
        return


    ret, frame = cap.read()


    if not ret:

        status_var.set(
            "Camera Frame Error"
        )

        status_label.config(
            fg=RED
        )

        root.after(
            30,
            update_camera
        )

        return


    # Mirror camera
    frame = cv2.flip(
        frame,
        1
    )


    # Convert to RGB
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # MediaPipe image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )


    # Detect hand
    result = landmarker.detect(
        mp_image
    )


    # ========================================================
    # HAND DETECTED
    # ========================================================

    if result.hand_landmarks:

        hand_landmarks = (
            result.hand_landmarks[0]
        )


        h, w, _ = frame.shape


        # Draw landmarks
        for landmark in hand_landmarks:

            x = int(
                landmark.x * w
            )

            y = int(
                landmark.y * h
            )


            cv2.circle(
                frame,
                (x, y),
                5,
                (0, 255, 0),
                -1
            )


        # Hand connections
        connections = [

            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),

            (0, 5),
            (5, 6),
            (6, 7),
            (7, 8),

            (0, 9),
            (9, 10),
            (10, 11),
            (11, 12),

            (0, 13),
            (13, 14),
            (14, 15),
            (15, 16),

            (0, 17),
            (17, 18),
            (18, 19),
            (19, 20),

            (5, 9),
            (9, 13),
            (13, 17)
        ]


        for start, end in connections:

            x1 = int(
                hand_landmarks[start].x * w
            )

            y1 = int(
                hand_landmarks[start].y * h
            )

            x2 = int(
                hand_landmarks[end].x * w
            )

            y2 = int(
                hand_landmarks[end].y * h
            )


            cv2.line(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )


        # Extract features
        features = extract_features(
            hand_landmarks
        )


        # Predict gesture
        prediction = model.predict(
            features
        )[0]


        # Confidence
        confidence = 0


        if hasattr(
            model,
            "predict_proba"
        ):

            probabilities = (
                model.predict_proba(
                    features
                )[0]
            )

            confidence = float(
                np.max(
                    probabilities
                ) * 100
            )


        prediction = str(
            prediction
        ).upper()


        # Update detection
        gesture_var.set(
            prediction
        )

        confidence_var.set(
            f"Confidence: {confidence:.1f}%"
        )

        confidence_bar["value"] = (
            confidence
        )


        status_var.set(
            "Hand Detected"
        )

        status_label.config(
            fg=GREEN
        )


        # Highlight card
        highlight_gesture(
            prediction
        )


        # Add history only when gesture changes
        if (
            len(prediction_history) == 0
            or
            prediction_history[0].split(
                "    "
            )[0] != prediction
        ):

            add_prediction_history(
                prediction,
                confidence
            )


        # Display prediction on camera
        cv2.putText(
            frame,
            prediction,
            (20, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            3
        )


    # ========================================================
    # NO HAND
    # ========================================================

    else:

        gesture_var.set(
            "NO HAND DETECTED"
        )

        confidence_var.set(
            "Confidence: 0%"
        )

        confidence_bar["value"] = 0


        status_var.set(
            "Waiting for Hand"
        )

        status_label.config(
            fg=SECONDARY
        )


        highlight_gesture("")


    # ========================================================
    # DISPLAY CAMERA
    # ========================================================

    display_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    image = Image.fromarray(
        display_frame
    )


    image = image.resize(
        (640, 430)
    )


    photo = ImageTk.PhotoImage(
        image=image
    )


    camera_label.config(
        image=photo,
        text=""
    )

    camera_label.image = photo


    if camera_running:

        root.after(
            30,
            update_camera
        )


# ============================================================
# EXIT
# ============================================================

def exit_application():

    global cap

    if cap is not None:

        cap.release()


    try:

        landmarker.close()

    except Exception:

        pass


    root.destroy()


# ============================================================
# BOTTOM BUTTONS
# ============================================================

button_frame = tk.Frame(
    root,
    bg=BG
)

button_frame.pack(
    fill="x",
    padx=20,
    pady=(0, 15)
)


start_button = tk.Button(
    button_frame,
    text="START",
    command=start_camera,
    font=("Arial", 11, "bold"),
    bg=GREEN,
    fg="white",
    activebackground=GREEN,
    activeforeground="white",
    relief="flat",
    width=15,
    height=2,
    cursor="hand2"
)

start_button.pack(
    side="left",
    padx=5
)


stop_button = tk.Button(
    button_frame,
    text="STOP",
    command=stop_camera,
    font=("Arial", 11, "bold"),
    bg="#555555",
    fg="white",
    activebackground="#555555",
    activeforeground="white",
    relief="flat",
    width=15,
    height=2,
    cursor="hand2"
)

stop_button.pack(
    side="left",
    padx=5
)


exit_button = tk.Button(
    button_frame,
    text="EXIT",
    command=exit_application,
    font=("Arial", 11, "bold"),
    bg="#8B0000",
    fg="white",
    activebackground="#8B0000",
    activeforeground="white",
    relief="flat",
    width=15,
    height=2,
    cursor="hand2"
)

exit_button.pack(
    side="right",
    padx=5
)


# ============================================================
# INITIAL STATE
# ============================================================

highlight_gesture("")


root.protocol(
    "WM_DELETE_WINDOW",
    exit_application
)


root.mainloop()