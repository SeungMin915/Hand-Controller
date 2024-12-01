import speech_recognition as sr
import pyautogui
import pyperclip
import beepsound

Recognizer = sr.Recognizer()
mic = sr.Microphone()

def input_text():
    beepsound.beepsound()
    with mic as source:
        audio = Recognizer.listen(source)
    try:
        data = Recognizer.recognize_google(audio, language="ko-KR")
        pyperclip.copy(data)
        pyautogui.hotkey('ctrl', 'v')
    except:
        print("이해하지 못했음")