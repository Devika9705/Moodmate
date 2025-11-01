import cv2
import numpy as np
import time
from tensorflow.keras.models import load_model
from collections import deque
import matplotlib.pyplot as plt
from ai_mentor import get_ai_advice
import pyttsx3
from moodmate import moodmate_assistant

# --------------------------
# Load Model & Config
# --------------------------
model = load_model("face_sentiment_model.h5")
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

emotion_emojis = {
    'Angry': "😠", 'Disgust': "🤢", 'Fear': "😨",
    'Happy': "😄", 'Neutral': "😐", 'Sad': "😢", 'Surprise': "😲"
}

color_map = {
    'Angry': (0, 0, 255), 'Disgust': (0, 128, 0), 'Fear': (128, 0, 128),
    'Happy': (0, 255, 255), 'Neutral': (255, 255, 255),
    'Sad': (255, 0, 0), 'Surprise': (0, 255, 0)
}

# Emotion tracking
emotion_history = []

cap = cv2.VideoCapture(0)
prev_time = 0

print("🎥 Starting Emotion Recognition... Press 'q' to stop and see report.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    overlay = frame.copy()
    overlay = cv2.GaussianBlur(overlay, (15, 15), 0)
    frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi = roi_gray.astype('float') / 255.0
        roi = np.expand_dims(roi, axis=0)
        roi = np.expand_dims(roi, axis=-1)

        prediction = model.predict(roi, verbose=0)
        max_index = np.argmax(prediction)
        label = emotion_labels[max_index]
        confidence = float(np.max(prediction))

        emotion_history.append(label)
        color = color_map[label]

        # Draw bounding box and label
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.rectangle(frame, (x, y - 35), (x + w, y), (0, 0, 0), -1)
        cv2.putText(frame, f"{label} {emotion_emojis[label]} ({int(confidence * 100)}%)",
                    (x + 5, y - 10), cv2.FONT_HERSHEY_DUPLEX, 0.8, color, 2)

        # Confidence bar
        bar_width = int(w * confidence)
        cv2.rectangle(frame, (x, y + h + 10), (x + bar_width, y + h + 20), color, -1)

    # FPS Counter
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {int(fps)}  |  Press 'Q' to exit", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.imshow("🧠 Real-Time Emotion Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --------------------------
# After Exit — Analysis
# --------------------------
cap.release()
cv2.destroyAllWindows()

print("\n📊 Generating Facial Expression Report...")

if len(emotion_history) == 0:
    print("No faces detected — no report to generate.")
else:
    unique, counts = np.unique(emotion_history, return_counts=True)
    total = len(emotion_history)
    percentages = (counts / total) * 100

    print("\n🧾 Emotion Summary Report:")
    for emo, pct in zip(unique, percentages):
        print(f"   {emotion_emojis[emo]} {emo}: {pct:.2f}%")

    # Bar Chart Visualization
    plt.figure(figsize=(8, 5))
    plt.bar(unique, percentages, color=[np.array(color_map[e])/255 for e in unique])
    plt.title("Facial Expression Analysis Report")
    plt.ylabel("Percentage (%)")
    plt.xlabel("Emotion Type")
    plt.grid(axis='y', linestyle='--', alpha=0.7)


    # Save report image
    plt.savefig("emotion_report.png")


    print("\n✅ Report saved as 'emotion_report.png'")
    plt.show()
# --------------------------
# Mentor Activation
# --------------------------
# Mentor Activation (Enhanced)
emotion = unique[np.argmax(percentages)]

def ai_mentor_response(emotion):
    """
    Returns motivational advice based on detected emotion.
    This version does not use the OpenAI API.
    """

    responses = {
        "Sad": "Hey, it’s okay to feel low sometimes. Take a deep breath and remember — you’re stronger than you think.",
        "Happy": "That’s amazing! Keep spreading that positive energy!",
        "Angry": "Try pausing and breathing. Calm minds make better choices.",
        "Surprise": "Wow! Life’s full of surprises — embrace them with curiosity!",
        "Fear": "Courage isn’t the absence of fear, it’s moving forward despite it. You’ve got this.",
        "Disgust": "It’s okay to dislike certain things. Let’s shift focus to what brings you peace.",
        "Neutral": "Let’s make today productive — one small step at a time!"
    }

    # Return matching advice or a default message
    return responses.get(emotion, "Stay positive and keep moving forward!")
# --------------------------
# Voice Mentor Activation
# --------------------------
def speak_with_emotion(advice, emotion):
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')

    # Prefer male voice if available
    male_voice = None
    for voice in voices:
        if "male" in voice.name.lower() or "daniel" in voice.name.lower() or "david" in voice.name.lower():
            male_voice = voice.id
            break
    if male_voice:
        engine.setProperty('voice', male_voice)

    # Emotion-based tone
    if emotion == "Sad":
        engine.setProperty('rate', 140)
        engine.setProperty('volume', 0.8)
    elif emotion == "Angry":
        engine.setProperty('rate', 170)
        engine.setProperty('volume', 1.0)
    elif emotion == "Happy":
        engine.setProperty('rate', 185)
        engine.setProperty('volume', 1.0)
    elif emotion == "Fear":
        engine.setProperty('rate', 150)
        engine.setProperty('volume', 0.9)
    else:
        engine.setProperty('rate', 160)
        engine.setProperty('volume', 0.9)

    print(f"\n🎤 Mentor speaking in {emotion.lower()} tone...")
    engine.say(advice)
    engine.runAndWait()
    engine.stop()


# --------------------------
# Run Mentor after report
# --------------------------
advice = get_ai_advice(emotion)
print(f"\n🧭 Detected dominant emotion: {emotion}")
print(f"💬 Mentor advice: {advice}")
speak_with_emotion(advice, emotion)
import threading

print("\n🪄 Activating MoodMate Assistant...")
moodmate_assistant(emotion)