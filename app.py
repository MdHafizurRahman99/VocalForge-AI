import gradio as gr
from TTS.api import TTS
import time

def load_tts_model():
    try:
        return TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", 
                 progress_bar=False, 
                 gpu=False)
    except Exception as e:
        print(f"Model loading error: {e}")
        return None

tts = load_tts_model()

def generate_speech(text):
    if not text.strip():
        return None, "Please enter some text"
    
    try:
        output_path = f"output_{int(time.time())}.wav"
        if tts:
            tts.tts_to_file(text=text, file_path=output_path)
            return output_path, "Success!"
        return None, "Model not loaded"
    except Exception as e:
        return None, f"Error: {str(e)}"

demo = gr.Interface(
    fn=generate_speech,
    inputs=gr.Textbox(label="Input Text", 
                    value="Hello! I'm converting text to speech."),
    outputs=[
        gr.Audio(label="Generated Speech"),
        gr.Textbox(label="Status")
    ],
    title="🎤 Text-to-Speech Converter",
    description="Convert text to speech using Coqui TTS"
)

demo.launch()