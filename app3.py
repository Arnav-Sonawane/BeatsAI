from flask import Flask, request, jsonify, send_file
from torch import torch
import soundfile as sf
import io
import numpy as np
from diffusers import StableAudioPipeline
from huggingface_hub import login
from spleeter.separator import Separator
import tempfile
import os

app = Flask(__name__)

# Pipeline Status Tracking
class PipelineStatus:
    def __init__(self):
        self.audio_gen_ready = False
        self.stem_sep_ready = False
        self.audio_gen_error = None


status = PipelineStatus()

# 1. Initialize Audio Generation Pipeline (GPU/CPU)
try:
    login(token="api_key")
    gen_pipe = StableAudioPipeline.from_pretrained(
        "stabilityai/stable-audio-open-1.0",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    ).to("cuda" if torch.cuda.is_available() else "cpu")
    status.audio_gen_ready = True
    print("? Audio Generation Pipeline Ready")
except Exception as e:
    status.audio_gen_error = str(e)
    print(f"? Audio Generation Failed: {str(e)}")

# 2. Initialize Stem Separation Pipeline (CPU)
try:
    stem_separator = Separator('spleeter:4stems')  # drums, bass, piano, other
    status.stem_sep_ready = True
    print("? Stem Separation Pipeline Ready")
except Exception as e:
    status.stem_sep_error = str(e)
    print(f"? Stem Separation Failed: {str(e)}")

# Health Check Endpoint
@app.route('/status')
def pipeline_status():
    return jsonify({
        "audio_generation": {
            "ready": status.audio_gen_ready,
            "error": status.audio_gen_error
        },
        "stem_separation": {
            "ready": status.stem_sep_ready,
            "error": status.stem_sep_error
        }
    })

# Audio Generation Only
@app.route('/generate', methods=['POST'])
def generate():
    if not status.audio_gen_ready:
        return jsonify({"error": "Audio generation unavailable", "details": status.audio_gen_error}), 503
    
    try:
        data = request.get_json()
        result = gen_pipe(
            prompt=data.get("prompt", "electronic music"),
            num_inference_steps=min(int(data.get("steps", 50)), 100)
        )
        audio_data = result.audios[0].T.cpu().numpy().astype(np.float32)
        
        buffer = io.BytesIO()
        sf.write(buffer, audio_data, 44100, format='WAV')
        buffer.seek(0)
        return send_file(buffer, mimetype='audio/wav')
    
    except Exception as e:
        return jsonify({"error": "Generation failed", "details": str(e)}), 500

@app.route('/split', methods=['POST'])
def split():
    if not status.stem_sep_ready:
        return jsonify({"error": "Stem separation unavailable"}), 503
    
    # Option 1: Accept file upload
    if 'file' in request.files:
        audio_file = request.files['file']
        tmp_path = os.path.join(tempfile.gettempdir(), audio_file.filename)
        audio_file.save(tmp_path)
    
    # Option 2: Use local file path (e.g., "saved_audio.wav" in same directory)
    elif 'file_path' in request.json:
        tmp_path = os.path.abspath(request.json['file_path'])
        if not os.path.exists(tmp_path):
            return jsonify({"error": "Local file not found"}), 404
    else:
        return jsonify({"error": "No audio provided (use 'file' or 'file_path')"}), 400
    
    try:
        # Process stems
        output_dir = tempfile.mkdtemp()
        stem_separator.separate_to_file(tmp_path, output_dir)
        
        # Prepare response
        base_name = os.path.basename(tmp_path).replace('.wav', '')
        stems = {
            'drums': f"/download/{base_name}_drums.wav",
            'bass': f"/download/{base_name}_bass.wav",
            'piano': f"/download/{base_name}_piano.wav",
            'other': f"/download/{base_name}_other.wav"
        }
        
        # Move stems to accessible location
        for stem in stems.keys():
            os.rename(
                f"{output_dir}/{base_name}/{stem}.wav",
                f"temp_{base_name}_{stem}.wav"
            )
        
        return jsonify({"stems": stems})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Combined Generation + Separation
@app.route('/generate_and_split', methods=['POST'])
def generate_and_split():
    # First generate audio
    gen_response = generate()
    if gen_response.status_code != 200:
        return gen_response  # Forward the error
    
    # Then separate stems (mock file upload)
    with tempfile.NamedTemporaryFile(suffix='.wav') as tmp:
        tmp.write(gen_response.data)
        tmp.seek(0)
        return split()

# File Download Endpoint
@app.route('/download/<filename>')
def download(filename):
    if not os.path.exists(f"temp_{filename}"):
        return jsonify({"error": "File not found"}), 404
    return send_file(f"temp_{filename}", mimetype='audio/wav')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, threaded=True)