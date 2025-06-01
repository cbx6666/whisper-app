import os
import wave
import contextlib
from flask import Flask, request, jsonify, render_template, send_from_directory
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import librosa
import numpy as np

app = Flask(__name__)

# 加载微调模型
MODEL_PATH = "/mnt/d/WSL/asr_large_model/web/merged_model"
processor = WhisperProcessor.from_pretrained(MODEL_PATH)
model = WhisperForConditionalGeneration.from_pretrained(MODEL_PATH)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    temp_path = None
    try:
        # 检查文件上传
        if 'audio' not in request.files:
            return jsonify({"error": "未上传音频文件"}), 400
        
        audio_file = request.files['audio']
        
        # 验证文件类型
        if not audio_file.filename.lower().endswith('.wav'):
            return jsonify({"error": "仅支持WAV格式文件"}), 400
        
        # 保存临时文件
        temp_path = "temp_audio.wav"
        audio_file.save(temp_path)
        
        # 验证WAV文件格式
        try:
            with contextlib.closing(wave.open(temp_path, 'r')) as wav_file:
                n_channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                frame_rate = wav_file.getframerate()
                n_frames = wav_file.getnframes()
                
                if n_channels != 1:
                    return jsonify({
                        "error": f"仅支持单声道音频，实际为{n_channels}声道",
                        "expected": 1,
                        "actual": n_channels
                    }), 400
                
                if frame_rate != 16000:
                    return jsonify({
                        "error": f"采样率应为16000Hz，实际为{frame_rate}Hz",
                        "expected": 16000,
                        "actual": frame_rate
                    }), 400
                
                if sample_width != 2:  # 16-bit
                    return jsonify({
                        "error": f"仅支持16-bit音频，实际为{sample_width*8}-bit",
                        "expected": 16,
                        "actual": sample_width * 8
                    }), 400
        except Exception as e:
            return jsonify({"error": f"无效的WAV文件: {str(e)}"}), 400
        
        # 加载音频进行识别
        y, sr = librosa.load(temp_path, sr=16000, mono=True)
        
        # 语音识别
        inputs = processor(y, sampling_rate=16000, return_tensors="pt")
        predicted_ids = model.generate(
            inputs.input_features,
            num_beams=5,
            temperature=0.2
        )
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        
        return jsonify({"text": transcription})
    except Exception as e:
        app.logger.error(f"识别错误: {str(e)}")
        return jsonify({"error": f"处理音频时发生错误: {str(e)}"}), 500
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True, debug=True)