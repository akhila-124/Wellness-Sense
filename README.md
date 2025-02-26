# Wellness-Sense
Wellness Sense is an AI-powered emotional health monitoring tool designed to detect signs of depression in under five minutes. The system leverages deep learning techniques to analyze facial expressions and audio cues using CNNs, alongside structured questionnaires and GPT-4o based assessments. By providing a non-intrusive and accessible method for early detection, this solution empowers individuals to take proactive steps toward mental well-being while offering personalized insights and treatment recommendations.
## Prerequisites
- Python
- OpenAI API key
## Installation 
1. Clone the repository <br>
`git clone https://github.com/akhila-124/Wellness-Sense.git`
2. Install required Python packages<br>
`pip install tensorflow keras opencv-python numpy openai websockets asyncio librosa soundfile matplotlib scikit-learn h5py python-dotenv`
3. Set up environment variables by creating a .env file in the project root<br>
`OPENAI_API_KEY=your_openai_api_key`<br>
`WS_HOST=127.0.0.1`<br>
`WS_PORT=8005`

## Running the Application
1. Start the backend server<br>
`python app.py`
2. Open `index.html` in your web browser to access the application

## Model Information
- `fer_saved_model.h5`: Pre-trained facial expression recognition model
- `ser_saved_model.h5`: Pre-trained speech emotion recognition model

## Dataset Information
- Facial expression recognition: FER2013 dataset
- Speech emotion recognition: RAVDESS dataset

## Training Your Own Models
- For facial expression recognition: Modify `fer_model.py` with the correct path to your dataset
- For speech emotion recognition: Update the file path in `ser_model.py` to point to your RAVDESS dataset
## Project Structure
- app.py: Main application server
- fer_model.py: Facial expression recognition model training script
- ser_model.py: Speech emotion recognition model training script
- chatscript.js: Frontend JavaScript for WebSocket communication
- index.html, details.html, chatbot.html, self.html: Frontend UI files
- Various CSS files for styling
