import pyttsx3
import speech_recognition as sr
import webbrowser
import time

def speak(text, rate=165):
    engine = pyttsx3.init()
    voices = engine.getProperty("voices")
    for v in voices:
        if "male" in v.name.lower() or "david" in v.name.lower():
            engine.setProperty("voice", v.id)
            break
    engine.setProperty("rate", rate)
    engine.say(text)
    engine.runAndWait()

def listen():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("🎤 Listening...")
        audio = recognizer.listen(source, phrase_time_limit=5)
    try:
        command = recognizer.recognize_google(audio)
        print(f"🗣️ You said: {command}")
        return command.lower()
    except sr.UnknownValueError:
        speak("Sorry, I didn’t catch that.")
        return ""
    except sr.RequestError:
        speak("Speech service is unavailable.")
        return ""

def moodmate_assistant(emotion):
    speak(f"I noticed you're feeling {emotion.lower()} today.")
    time.sleep(1)
    speak("Would you like to change your mood or stay in it?")
    user_response = listen()

    if "change" in user_response:
        speak("Alright! Let’s change that vibe.")
        if emotion == "sad":
            webbrowser.open("https://www.youtube.com/watch?v=ZbZSe6N_BXs")
        elif emotion == "angry":
            webbrowser.open("https://www.youtube.com/watch?v=ktvTqknDobU")
        elif emotion == "neutral":
            webbrowser.open("https://www.youtube.com/watch?v=1y6smkh6c-0")
        else:
            webbrowser.open("https://www.youtube.com/watch?v=2Vv-BfVoq4g")
        speak("Here’s something to lift your mood! 🎵")
    elif "stay" in user_response:
        speak("Got it. Let’s keep it peaceful and steady.")
        webbrowser.open("https://www.youtube.com/watch?v=jfKfPfyJRdk")
    else:
        speak("Alright, I’ll just stay with you. Remember, every feeling is valid 💛")
