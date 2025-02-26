import asyncio
import json
import websockets
import time
from openai import OpenAI
import librosa
import soundfile
import numpy as np
from keras.models import load_model
import tkinter as tk
import tkinter.messagebox as messagebox
import sounddevice as sd
import soundfile as sf
import pyttsx3
import os
import cv2
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from dotenv import load_dotenv
load_dotenv()

key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=key)

HOST = os.getenv("WS_HOST", "0.0.0.0")  # Default to 0.0.0.0 (for external access)
PORT = int(os.getenv("WS_PORT", 8005))  # Default port 8005

# Function to handle WebSocket connections from the JavaScript client
async def handle_connection(websocket, path):
    print('Connected to JS')

    while True:
        data = await websocket.recv()
        if not data:
            break  

        re=data.split('==')
        print()
        await process_data(data, websocket)


# Function to process incoming data from the WebSocket connection
async def process_data(websocket):
    data = await websocket.recv()
    print(data)
    re=data.split('==')

    # Handling different types of requests
    if re[0].lower() == 'medical prescription':
        response={
            "msg" : "Please enter your current problems",
            "option" : [],
            "nextEvent" : 'LLMcall',
        }
        await send_response(response, websocket)
        user_response = await asyncio.wait_for(websocket.recv(), timeout=100)
        print('++++++++++++++',user_response)
        temp=user_response.split('==')
        print("{}('{}', websocket)".format(temp[3], user_response))
        await eval("{}('{}', websocket)".format(temp[3], user_response))
    elif re[0].lower() == 'selfcareactivities':
        print('in selfcare activities')
        await LLMcall_Act(data,websocket)
    elif re[0].lower() == 'prescription':
        print('in prescription')
        await LLMcall_pres(data,websocket)
    elif re[0].lower() == 'voice based':
        print('voice based')
        await voice(data,websocket)
        
    elif re[0].lower() == 'depression test':
        face_value=camcall()
        response={
            "msg" : "I had trouble relaxing and calming down.",
            "option" : ["1- Did not apply to me at all",
                        "2- Applied to me to some degree, or some of the time",
                        "3- Applied to me to a considerable degree or a good part of time",
                        "4- Applied to me very much or most of the time"],
            "nextEvent" : 'q1',
            "cnt" : 0,
            "face_value":face_value
        }
        await send_response(response, websocket)
        user_response = await asyncio.wait_for(websocket.recv(), timeout=100)
        print('++++++++++++++',user_response)
        temp=user_response.split('==')
        print(temp)
        print("{}('{}', websocket)".format(temp[3], user_response))
        await eval("{}('{}', websocket)".format(temp[3], user_response))
    else:
        await default_response(websocket)


# Function to handle voice-based emotion detection
async def voice(data,websocket):
 
    # Extract features from an audio file for emotion recognition
    def extract_feature(file_name, mfcc, chroma, mel):
        with soundfile.SoundFile(file_name) as sound_file:
            X = sound_file.read(dtype="float32")
            sample_rate = sound_file.samplerate
            if chroma:
                stft = np.abs(librosa.stft(X))
            result = np.array([])
            if mfcc:
                mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
                result = np.hstack((result, mfccs))
            if chroma:
                chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
                result = np.hstack((result, chroma))
            if mel:
                mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
                result = np.hstack((result, mel))
            return result

  # Emotion labels based on dataset encoding
    emotions = {
        '01': 'neutral',
        '02': 'calm',
        '03': 'happy',
        '04': 'sad',
        '05': 'angry',
        '06': 'fearful',
        '07': 'disgust',
        '08': 'surprised'
    }


    print('loading the model')


    # Load pre-trained Speech Emotion Recognition (SER) model
    model = load_model('ser_saved_model.h5')


    # Function to predict emotion using the trained CNN model
    def predict_emotion_cnn(file_path):
        features = extract_feature(file_path, mfcc=True, chroma=True, mel=True)

        # Reshape the features to match the shape expected by the CNN model
        features = features.reshape(1, -1, 1)  # Adjust the shape based on your model input shape

        predicted_emotion = model.predict(features)

        print(predicted_emotion)
        print(np.argmax(predicted_emotion))

        predicted_emotion_index = np.argmax(predicted_emotion)
       
        ans = ''
        if predicted_emotion_index == 4 or predicted_emotion_index == 5 or predicted_emotion_index == 6 or predicted_emotion_index == 7:
            ans = 'depressed'
        else:
            ans = 'normal'

        return ans


    def chatbot(emotion):
        if emotion == "normal":
            return "You sound normal. Have a nice day and relax."
        else:
            return "You sound depressed. It's okay to feel down sometimes. Reach out to someone for support."


    # Define the GUI class
    class EmotionPredictionGUI:
        def __init__(self):
            self.window = tk.Tk()
            self.window.title("Depression Detection")
            self.window.geometry("300x320")

            self.window.iconbitmap('psychologyicon.ico')


            # Load the image
            background_image = tk.PhotoImage(file="background.png")  # Replace "background_image.png" with your image file

            # Get the width and height of the window
            window_width = self.window.winfo_width()
            window_height = self.window.winfo_height()

            # Create a Label widget to hold the image
            background_label = tk.Label(self.window, image=background_image)
            background_label.place(x=0, y=0, relwidth=1, relheight=1)

            label = tk.Label(background_label, text="Hello, Please tell us about your day", font=("Helvetica", 12))

            # Pack the Label widget to display it in the window
            label.pack(pady=30)
            # Record button
            self.record_button = tk.Button(background_label, text="Speak", command=self.record_speech, bg="#14c6d3", fg="black",
                                        font=("Helvetica", 14), relief="raised")
            self.record_button.pack(pady=50)
    

            self.engine = pyttsx3.init()
            self.engine.setProperty("rate", 150)

            self.window.mainloop()

            self.engine.runAndWait()

        # Function to record speech
        def record_speech(self):
            # Configure the recording parameters
            sample_rate = 16000  # Sample rate of the audio
            duration = 5  # Duration of the recording in seconds

            # Record the speech
            messagebox.showinfo("Recording", "Recording started. Please speak for {} seconds.".format(duration))
            audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
            sd.wait()

            # Save the recorded speech to a file
            file_path = "recorded_speech.wav"
            sf.write(file_path, audio, sample_rate)

            # Predict the emotion for the recorded speech
            predicted_emotion = predict_emotion_cnn(file_path)

            # Show the predicted emotion in a message box

            response = chatbot(predicted_emotion)

            messagebox.showinfo("Depression Detection", "Predicted Emotion: {}".format(predicted_emotion))

            self.engine.say(response)
            self.engine.runAndWait()

    gui = EmotionPredictionGUI()



async def LLMcall(data,websocket):
    print(data)
    req=data.split('==')
    question = '{}, strictly suggest me with appropriate medicine or recovery procedure for a person with age:{} and gender:{}, the response has to be in a proper format with no more than 150 tokens'.format(req[0],req[1],req[2])
    prompt = f"Question: {question}\nAnswer:"
    response = client.chat.completions.create(model="gpt-4o",  
    messages=[{"role": "system", "content": prompt}])
    answer = response.choices[0].message.content
    response={
        'ans' : answer,
        'nextEvent' : 'endMed'
    }
    print(answer)
    print('+++++++++LLM Response Generated')
    await send_response(response, websocket)

async def LLMcall_Act(data,websocket):
    print(data)
    req=data.split('==')
    question = 'DASS score is: {}, strictly suggest me with appropriate self-care activities to come out of depression for a person with age:{} and gender:{} based on DASS score given, the response has to be in a proper format as 10 points '.format(req[3],req[1],req[2])
    prompt = f"Question: {question}\nAnswer:"
    response = client.chat.completions.create(model="gpt-4o",  
    messages=[{"role": "system", "content": prompt}])
    answer = response.choices[0].message.content
    response={
        'ans' : answer,
        'nextEvent' : 'endAct'
    }
    print(answer)
    print('+++++++++LLM Response Generated')
    await send_response(response, websocket)

    
async def LLMcall_pres(data,websocket):
    print(data)
    req=data.split('==')
    question = 'DASS score is: {}, strictly suggest me with appropriate medical prescription,to come out of depression for a person with age:{} and gender:{} based on DASS score given, the response has to be in a proper format not more than 150 tokens '.format(req[3],req[1],req[2])
    prompt = f"Question: {question}\nAnswer:"
    response = client.chat.completions.create(model="gpt-4o",  
    messages=[{"role": "system", "content": prompt}])
    answer = response.choices[0].message.content
    response={
        'ans' : answer,
        'nextEvent' : 'endPres'
    }
    print(answer)
    print('+++++++++LLM Response Generated')
    await send_response(response, websocket)



async def q1(data, websocket):
    print(data)
    opt=data.split('==')
    req=opt[0].split('-')
    print(int(req[0]))
    face_value=camcall()
    response={
            "msg" : "I was aware of dryness of my mouth.",
            "option" : ["1- Did not apply to me at all",
                        "2- Applied to me to some degree, or some of the time",
                        "3- Applied to me to a considerable degree or a good part of time",
                        "4- Applied to me very much or most of the time"],
            "nextEvent" : 'q2',
            "cnt" : int(req[0]),
            "face_value":face_value
        }
    print('=====================','Reached here')
    
    await send_response(response, websocket)
    user_response = await asyncio.wait_for(websocket.recv(), timeout=100)
    print('++++++++++++++',user_response)
    temp=user_response.split('==')
    print("{}('{}', websocket)".format(temp[3], user_response))
    await eval("{}('{}', websocket)".format(temp[3], user_response))

async def q2(data, websocket):
    print(data)
    opt=data.split('==')
    req=opt[0].split('-')
    print(int(req[0]))
    face_value=camcall()
    response={
            "msg" : "I couldn’t seem to experience any positive feeling at all",
            "option" : ["1- Did not apply to me at all",
                        "2- Applied to me to some degree, or some of the time",
                        "3- Applied to me to a considerable degree or a good part of time",
                        "4- Applied to me very much or most of the time"],
            "nextEvent" : 'q3',
            "cnt" : int(req[0]),
            "face_value":face_value
        }
    print('=====================','Reached here')
    await send_response(response, websocket)
    user_response = await asyncio.wait_for(websocket.recv(), timeout=100)
    print('++++++++++++++',user_response)
    temp=user_response.split('==')
    print("{}('{}', websocket)".format(temp[3], user_response))
    await eval("{}('{}', websocket)".format(temp[3], user_response))

async def q3(data, websocket):
    print(data)
    opt=data.split('==')
    req=opt[0].split('-')
    print(int(req[0]))
    face_value=camcall()
    response={
            "msg" : "I experienced breathing difficulty (e.g. excessively rapid breathing, breathlessness in the absence of physical exertion) ",
            "option" : ["1- Did not apply to me at all",
                        "2- Applied to me to some degree, or some of the time",
                        "3- Applied to me to a considerable degree or a good part of time",
                        "4- Applied to me very much or most of the time"],
            "nextEvent" : 'q4',
            "cnt" : int(req[0]),
            "face_value":face_value
        }
    print('=====================','Reached here q3')
    await send_response(response, websocket)
    user_response = await asyncio.wait_for(websocket.recv(), timeout=100)
    print('++++++++++++++',user_response)
    temp=user_response.split('==')
    print("{}('{}', websocket)".format(temp[3], user_response))
    await eval("{}('{}', websocket)".format(temp[3], user_response))

async def q4(data, websocket):
    print(data)
    opt=data.split('==')
    req=opt[0].split('-')
    face_value=camcall()
    print(int(req[0]))
    response={
            "msg" : "I found it difficult to work up the initiative to do things",
            "option" : ["1- Did not apply to me at all",
                        "2- Applied to me to some degree, or some of the time",
                        "3- Applied to me to a considerable degree or a good part of time",
                        "4- Applied to me very much or most of the time"],
            "nextEvent" : 'q5',
            "cnt" : int(req[0]),
            "face_value":face_value
        }
    print('=====================','Reached here')
    await send_response(response, websocket)
    user_response = await asyncio.wait_for(websocket.recv(), timeout=100)
    print('++++++++++++++',user_response)
    temp=user_response.split('==')
    print("{}('{}', websocket)".format(temp[3], user_response))
    await eval("{}('{}', websocket)".format(temp[3], user_response))

async def q5(data, websocket):
    print(data)
    opt=data.split('==')
    req=opt[0].split('-')
    face_value=camcall()
    response={
            "msg" : "I tended to over-react to situations",
            "option" : ["1- Did not apply to me at all",
                        "2- Applied to me to some degree, or some of the time",
                        "3- Applied to me to a considerable degree or a good part of time",
                        "4- Applied to me very much or most of the time"],
            "nextEvent" : 'q6',
            "cnt" : int(req[0]),
            "face_value":face_value
        }
    print('=====================','Reached here q5')
    await send_response(response, websocket)
    user_response = await asyncio.wait_for(websocket.recv(), timeout=100)
    print('++++++++++++++',user_response)
    temp=user_response.split('==')
    print("{}('{}', websocket)".format(temp[3], user_response))
    await eval("{}('{}', websocket)".format(temp[3], user_response))

async def q6(data, websocket):
    print(data)
    opt=data.split('==')
    req=opt[0].split('-')
    print(int(req[0]))
    face_value=camcall()
    response={
            "msg" : "I experienced trembling (e.g. in the hands)",
            "option" : ["1- Did not apply to me at all",
                        "2- Applied to me to some degree, or some of the time",
                        "3- Applied to me to a considerable degree or a good part of time",
                        "4- Applied to me very much or most of the time"],
            "nextEvent" : 'q7',
            "cnt" : int(req[0]),
            "face_value":face_value
        }
    print('=====================','Reached here q6')
    await send_response(response, websocket)
    user_response = await asyncio.wait_for(websocket.recv(), timeout=100)
    print('++++++++++++++',user_response)
    temp=user_response.split('==')
    print("{}('{}', websocket)".format(temp[3], user_response))
    await eval("{}('{}', websocket)".format(temp[3], user_response))

async def q7(data, websocket):
    print(data)
    opt=data.split('==')
    req=opt[0].split('-')
    face_value=camcall()
    response={
            "msg" : "I felt that I was using a lot of nervous energy",
            "option" : ["1- Did not apply to me at all",
                        "2- Applied to me to some degree, or some of the time",
                        "3- Applied to me to a considerable degree or a good part of time",
                        "4- Applied to me very much or most of the time"],
            "nextEvent" : 'q8',
            "cnt" : int(req[0]),
            "face_value":face_value
        }
    print('=====================','Reached here q7')
    await send_response(response, websocket)
    user_response = await asyncio.wait_for(websocket.recv(), timeout=100)
    print('++++++++++++++',user_response)
    temp=user_response.split('==')
    print("{}('{}', websocket)".format(temp[3], user_response))
    await eval("{}('{}', websocket)".format(temp[3], user_response))

async def q8(data, websocket):
    print(data)
    opt=data.split('==')
    face_value=camcall()
    req=opt[0].split('-')
    print(int(req[0]))
    response={
            "msg" : "I was worried about situations in which I might panic and make a fool of myself",
            "option" : ["1- Did not apply to me at all",
                        "2- Applied to me to some degree, or some of the time",
                        "3- Applied to me to a considerable degree or a good part of time",
                        "4- Applied to me very much or most of the time"],
            "nextEvent" : 'q9',
            "cnt" : int(req[0]),
            "face_value":face_value
        }
    print('=====================','Reached here q8')
    await send_response(response, websocket)
    user_response = await asyncio.wait_for(websocket.recv(), timeout=100)
    print('++++++++++++++',user_response)
    temp=user_response.split('==')
    print("{}('{}', websocket)".format(temp[3], user_response))
    await eval("{}('{}', websocket)".format(temp[3], user_response))

async def q9(data, websocket):
    print(data)
    opt=data.split('==')
    req=opt[0].split('-')
    face_value=camcall()
    response={
            "msg" : " I felt that I had nothing to look forward to",
            "option" : ["1- Did not apply to me at all",
                        "2- Applied to me to some degree, or some of the time",
                        "3- Applied to me to a considerable degree or a good part of time",
                        "4- Applied to me very much or most of the time"],
            "nextEvent" : 'q10',
            "cnt" : int(req[0]),
            "face_value":face_value
        }
    print('=====================','Reached here q9')
    await send_response(response, websocket)
    user_response = await asyncio.wait_for(websocket.recv(), timeout=100)
    print('++++++++++++++',user_response)
    temp=user_response.split('==')
    print("{}('{}', websocket)".format(temp[3], user_response))
    await eval("{}('{}', websocket)".format(temp[3], user_response))

async def q10(data, websocket):
    print(data)
    opt=data.split('==')
    req=opt[0].split('-')
    print(int(req[0]))
    face_value=camcall()
    response={
            "msg" : "I found myself getting troubled",
            "option" : ["1- Did not apply to me at all",
                        "2- Applied to me to some degree, or some of the time",
                        "3- Applied to me to a considerable degree or a good part of time",
                        "4- Applied to me very much or most of the time"],
            "nextEvent" : 'q11',
            "cnt" : int(req[0]),
            "face_value":face_value
        }
    print('=====================','Reached here q11')
    await send_response(response, websocket)
    user_response = await asyncio.wait_for(websocket.recv(), timeout=100)
    print('++++++++++++++',user_response)
    temp=user_response.split('==')
    print("{}('{}', websocket)".format(temp[3], user_response))
    await eval("{}('{}', websocket)".format(temp[3], user_response))

async def q11(data, websocket):
    print(data)
    opt=data.split('==')
    req=opt[0].split('-')
    face_value=camcall()
    response={
            "msg" : "I found it difficult to relax",
            "option" : ["1- Did not apply to me at all",
                        "2- Applied to me to some degree, or some of the time",
                        "3- Applied to me to a considerable degree or a good part of time",
                        "4- Applied to me very much or most of the time"],
            "nextEvent" : 'q12',
            "cnt" : int(req[0]),
            "face_value":face_value
        }
    print('=====================','Reached here q12')
    await send_response(response, websocket)
    user_response = await asyncio.wait_for(websocket.recv(), timeout=100)
    print('++++++++++++++',user_response)
    temp=user_response.split('==')
    print("{}('{}', websocket)".format(temp[3], user_response))
    await eval("{}('{}', websocket)".format(temp[3], user_response))

async def q12(data, websocket):
    print(data)
    opt=data.split('==')
    req=opt[0].split('-')
    print(int(req[0]))
    face_value=camcall()
    response={
            "msg" : "I felt down-hearted and unhappy",
            "option" : ["1- Did not apply to me at all",
                        "2- Applied to me to some degree, or some of the time",
                        "3- Applied to me to a considerable degree or a good part of time",
                        "4- Applied to me very much or most of the time"],
            "nextEvent" : 'q13',
            "cnt" : int(req[0]),
            "face_value":face_value
        }
    print('=====================','Reached here q12')
    await send_response(response, websocket)
    user_response = await asyncio.wait_for(websocket.recv(), timeout=100)
    print('++++++++++++++',user_response)
    temp=user_response.split('==')
    print("{}('{}', websocket)".format(temp[3], user_response))
    await eval("{}('{}', websocket)".format(temp[3], user_response))

async def q13(data, websocket):
    print(data)
    opt=data.split('==')
    face_value=camcall()
    req=opt[0].split('-')
    print(int(req[0]))
    response={
            "msg" : "I was intolerant of anything that kept me from getting on with what I was doing",
            "option" : ["1- Did not apply to me at all",
                        "2- Applied to me to some degree, or some of the time",
                        "3- Applied to me to a considerable degree or a good part of time",
                        "4- Applied to me very much or most of the time"],
            "nextEvent" : 'q14',
            "cnt" : int(req[0]),
            "face_value":face_value
        }
    print('=====================','Reached here q13')
    await send_response(response, websocket)
    user_response = await asyncio.wait_for(websocket.recv(), timeout=100)
    print('++++++++++++++',user_response)
    temp=user_response.split('==')
    print("{}('{}', websocket)".format(temp[3], user_response))
    await eval("{}('{}', websocket)".format(temp[3], user_response))

async def q14(data, websocket):
    print(data)
    opt=data.split('==')
    req=opt[0].split('-')
    print(int(req[0]))  
    face_value=camcall()
    response={
            "msg" : "I felt I was close to panic",
            "option" : ["1- Did not apply to me at all",
                        "2- Applied to me to some degree, or some of the time",
                        "3- Applied to me to a considerable degree or a good part of time",
                        "4- Applied to me very much or most of the time"],
            "nextEvent" : 'q15',
            "cnt" : int(req[0]),
            "face_value":face_value
        }
    print('=====================','Reached here q14')
    await send_response(response, websocket)
    user_response = await asyncio.wait_for(websocket.recv(), timeout=100)
    print('++++++++++++++',user_response)
    temp=user_response.split('==')
    print("{}('{}', websocket)".format(temp[3], user_response))
    await eval("{}('{}', websocket)".format(temp[3], user_response))

async def q15(data, websocket):
    print(data)
    opt=data.split('==')
    req=opt[0].split('-')
    face_value=camcall()
    print(int(req[0]))  
    response={
            "msg" : "I was unable to become enthusiastic about anything",
            "option" : ["1- Did not apply to me at all",
                        "2- Applied to me to some degree, or some of the time",
                        "3- Applied to me to a considerable degree or a good part of time",
                        "4- Applied to me very much or most of the time"],
            "nextEvent" : 'q16',
            "cnt" : int(req[0]),
            "face_value":face_value
        }
    print('=====================','Reached here q15')
    await send_response(response, websocket)
    user_response = await asyncio.wait_for(websocket.recv(), timeout=100)
    print('++++++++++++++',user_response)
    temp=user_response.split('==')
    print("{}('{}', websocket)".format(temp[3], user_response))
    await eval("{}('{}', websocket)".format(temp[3], user_response))

async def q16(data, websocket):
    print(data)
    opt=data.split('==')
    req=opt[0].split('-')
    print(int(req[0]))  
    face_value=camcall()
    response={
            "msg" : "I felt I wasn’t worth much as a person",
            "option" : ["1- Did not apply to me at all",
                        "2- Applied to me to some degree, or some of the time",
                        "3- Applied to me to a considerable degree or a good part of time",
                        "4- Applied to me very much or most of the time"],
            "nextEvent" : 'q17',
            "cnt" : int(req[0]),
            "face_value":face_value
        }
    print('=====================','Reached here q16')
    await send_response(response, websocket)
    user_response = await asyncio.wait_for(websocket.recv(), timeout=100)
    print('++++++++++++++',user_response)
    temp=user_response.split('==')
    print("{}('{}', websocket)".format(temp[3], user_response))
    await eval("{}('{}', websocket)".format(temp[3], user_response))

async def q17(data, websocket):
    print(data)
    opt=data.split('==')
    req=opt[0].split('-')
    print(int(req[0]))  
    face_value=camcall()
    response={
            "msg" : "I thought I was quite sensitive",
            "option" : ["1- Did not apply to me at all",
                        "2- Applied to me to some degree, or some of the time",
                        "3- Applied to me to a considerable degree or a good part of time",
                        "4- Applied to me very much or most of the time"],
            "nextEvent" : 'q18',
            "cnt" : int(req[0]),
            "face_value":face_value
        }
    print('=====================','Reached here q17')
    await send_response(response, websocket)
    user_response = await asyncio.wait_for(websocket.recv(), timeout=100)
    print('++++++++++++++',user_response)
    temp=user_response.split('==')
    print("{}('{}', websocket)".format(temp[3], user_response))
    await eval("{}('{}', websocket)".format(temp[3], user_response))

async def q18(data, websocket):
    print(data)
    opt=data.split('==')
    req=opt[0].split('-')
    face_value=camcall()
    response={
            "msg" : "I was aware of the action of my heart in the absence of physical exertion (e.g. sense of heart rate increase, heart missing a beat)",
            "option" : ["1- Did not apply to me at all",
                        "2- Applied to me to some degree, or some of the time",
                        "3- Applied to me to a considerable degree or a good part of time",
                        "4- Applied to me very much or most of the time"],
            "nextEvent" : 'q19',
            "cnt" : int(req[0]),
            "face_value":face_value
        }
    print('=====================','Reached here q18')
    await send_response(response, websocket)
    user_response = await asyncio.wait_for(websocket.recv(), timeout=100)
    print('++++++++++++++',user_response)
    temp=user_response.split('==')
    print("{}('{}', websocket)".format(temp[3], user_response))
    await eval("{}('{}', websocket)".format(temp[3], user_response))

async def q19(data, websocket):
    print(data)
    opt=data.split('==')
    req=opt[0].split('-')
    face_value=camcall()
    response={
            "msg" : "I felt scared without any good reason",
            "option" : ["1- Did not apply to me at all",
                        "2- Applied to me to some degree, or some of the time",
                        "3- Applied to me to a considerable degree or a good part of time",
                        "4- Applied to me very much or most of the time"],
            "nextEvent" : 'q20',
            "cnt" : int(req[0]),
            "face_value":face_value
        }
    print('=====================','Reached here q19')
    await send_response(response, websocket)
    user_response = await asyncio.wait_for(websocket.recv(), timeout=100)
    print('++++++++++++++',user_response)
    temp=user_response.split('==')
    print("{}('{}', websocket)".format(temp[3], user_response))
    await eval("{}('{}', websocket)".format(temp[3], user_response))

async def q20(data, websocket):
    print(data)
    opt=data.split('==')
    req=opt[0].split('-')
    face_value=camcall()
    response={
            "msg" : "I felt that life was meaningless ",
            "option" : ["1- Did not apply to me at all",
                        "2- Applied to me to some degree, or some of the time",
                        "3- Applied to me to a considerable degree or a good part of time",
                        "4- Applied to me very much or most of the time"],
            "nextEvent" : 'q21',
            "cnt" : int(req[0]),
            "face_value":face_value
        }
    print('=====================','Reached here q20')
    await send_response(response, websocket)
    user_response = await asyncio.wait_for(websocket.recv(), timeout=100)
    print('++++++++++++++',user_response)
    temp=user_response.split('==')
    print("{}('{}', websocket)".format(temp[3], user_response))
    await eval("{}('{}', websocket)".format(temp[3], user_response))

async def q21(data, websocket):
    print(data)
    opt=data.split('==')
    req=opt[0].split('-')
    face_value=camcall()
    response={
            "msg" : "I tended to over-react to situations",
            "option" : ["1- Did not apply to me at all",
                        "2- Applied to me to some degree, or some of the time",
                        "3- Applied to me to a considerable degree or a good part of time",
                        "4- Applied to me very much or most of the time"],
            "nextEvent" : 'end',
            "cnt" : int(req[0]),
            "face_value":face_value
        }
    print('=====================','Reached here q21')
    await send_response(response, websocket)
    user_response = await asyncio.wait_for(websocket.recv(), timeout=100)
    print('++++++++++++++',user_response)
    temp=user_response.split('==')
    print("{}('{}', websocket)".format(temp[3], user_response))
    await eval("{}('{}', websocket)".format(temp[3], user_response))


async def end(data,websocket):
    print(data)
    face_value=camcall()
    opt=data.split('==')
    req=opt[0].split('-')
    response={
        'nextEvent':'endDep',
        'cnt' : int(req[0]),
        "face_value":face_value

    }
    print('=====================','Dep Test Finished')
    await send_response(response, websocket)

async def default_response(websocket):
    print('Default response')
    response = {"msg": "I didn't understand that. Please choose a valid option.", "option": []}
    await send_response(response, websocket)


def camcall():
    cap = cv2.VideoCapture(0)

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))
    face_v=0
    count_d=0
    count_n=0
    if cap.isOpened() :
        model.load_weights('fer_saved_model.h5')
        start_time = time.time()


        while (time.time() - start_time) < 2.5:
            ret, frame = cap.read()
            if not ret:
                break
            facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
                roi_gray = gray[y:y + h, x:x + w]
                cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
                prediction = model.predict(cropped_img)
                maxindex = int(np.argmax(prediction))
                temp = ''
                face_v=0
                if maxindex == 0 or maxindex == 1 or maxindex == 2 or maxindex == 5:
                    temp = 'Depressed'
                    count_d+=1
                else:
                    temp = 'Normal'
                    count_n+=1
                cv2.putText(frame, temp, (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow('Video', cv2.resize(frame, (1600, 960), interpolation=cv2.INTER_CUBIC))
            cv2.setWindowProperty('Video', cv2.WND_PROP_TOPMOST, 1)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        if((count_d+count_n)!=0):
            face_v=(count_d*100)/(count_d+count_n)
        print("face_value:",face_v)
        return face_v
    
async def send_response(response, websocket):
    response_json = json.dumps(response)
    await websocket.send(response_json)

async def main():
    server = await websockets.serve(
        process_data, HOST, PORT
    )

    async with server:
        await server.wait_closed()

if __name__ == '__main__':
     asyncio.run(main())