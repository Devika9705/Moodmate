🎭 MoodMate: Real-Time Facial Emotion Detection with AI Mentor

**MoodMate** is an intelligent real-time facial emotion recognition system that detects human emotions through a webcam feed and provides instant AI-generated advice or responses tailored to your mood.  
Powered by deep learning and integrated with the **MoodMate Assistant (OpenAI)**, this app blends emotion-aware computing with an empathetic AI companion.

---

🚀 Features

- 🧠 **Real-Time Emotion Detection** – Detects emotions like *happy, sad, angry, neutral, surprised,* and more using your webcam.  
- 💬 **MoodMate AI Assistant** – Generates thoughtful advice, motivation, or comfort messages based on your detected mood.  
- 🌈 **Stylish UI** – Smooth and interactive interface with emotion overlay and live confidence display.  
- 🔊 **Voice Support (optional)** – Speaks AI-generated advice for an immersive experience.  
- ⚡ **Lightweight & Fast** – Uses OpenCV and deep learning with minimal latency.

---

## 🧩 Tech Stack

| Category | Tools & Libraries |
|-----------|-------------------|
| **Programming Language** | Python |
| **Deep Learning** | TensorFlow / Keras |
| **Computer Vision** | OpenCV |
| **AI Assistant** | OpenAI API |
| **Frontend (optional)** | Streamlit or OpenCV UI overlays |
| **Utilities** | NumPy, pyttsx3, cv2, requests |

---

## 🖥️ Project Structure
facial-expression-recognition/
│
├── ai_mentor.py # AI advice generator using OpenAI API
├── moodmate.py # MoodMate assistant module
├── realtime_emotion_stylish.py # Real-time webcam emotion detector with UI
├── realtime_emotion.py # Basic emotion detection script
├── test_camera.py # Camera test utility
├── app.py # Optional integration script
├── main.py # Main entry point
└── .gitignore

🎥 How It Works
The webcam captures your live facial expression.

The trained model predicts your emotion in real time.

The detected emotion is passed to MoodMate AI, which generates personalized advice.

The system optionally speaks the advice aloud for a realistic assistant feel.
| Emotion      | Example AI Response                                                                         |
| ------------ | ------------------------------------------------------------------------------------------- |
| 😊 Happy     | “That’s a wonderful smile! Keep spreading positivity.”                                      |
| 😞 Sad       | “It’s okay to feel low sometimes. Take a deep breath — you’re doing better than you think.” |
| 😠 Angry     | “Try taking a short walk or a few deep breaths. You deserve peace.”                         |
| 😮 Surprised | “Wow! Something caught you off guard? Life’s full of surprises!”                            |
| 😐 Neutral   | “Calm and balanced — a great state to focus your mind.”                                     |



🧾 Future Enhancements
🪄 Emotion-based music recommendation

🕹️ Integration with virtual avatars

📈 Mood tracking dashboard

🧬 Model fine-tuning for better emotion accuracy


