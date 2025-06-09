import os
import wave  # 用于处理WAV音频文件
import contextlib
from flask import Flask, request, jsonify, render_template, send_from_directory  # Web应用框架
import ctranslate2  # 优化的推理引擎
from transformers import WhisperProcessor  # Whisper模型处理器
import librosa  # 音频处理
import numpy as np  # 数值计算
from funasr import AutoModel

# 初始化Flask应用
app = Flask(__name__)

# 模型路径配置
MODEL_PATH = "/mnt/d/WSL/asr_large_model/web/merged_model"  # HuggingFace格式模型
CTRANSLATE_MODEL_PATH = "/mnt/d/WSL/asr_large_model/web/ctranslate_model"  # CTranslate2格式模型
punc_model = AutoModel(model="ct-punc", model_revision="v2.0.4")

# 加载模型组件
processor = WhisperProcessor.from_pretrained(MODEL_PATH)  # 加载特征提取器和分词器
translator = ctranslate2.models.Whisper(
    CTRANSLATE_MODEL_PATH, 
    device="cpu",  # 使用CPU进行推理
    compute_type="float32",  # CPU模式下使用float32以获得更好的兼容性
    intra_threads=4,  # 限制线程数以减少内存使用
    inter_threads=1  # 单个批次处理
)

# 首页路由
@app.route('/')
def index():
    return render_template('index.html')

# 静态文件路由
@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

# 音频转录API端点
@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    temp_path = None
    try:
        # 检查是否上传了音频文件
        if 'audio' not in request.files:
            return jsonify({"error": "未上传音频文件"}), 400
        
        audio_file = request.files['audio']
        # 获取前端传递的模式（chinese 或 english），默认为 chinese
        mode = request.form.get('mode', 'chinese')
        
        # 验证文件格式是否为WAV
        if not audio_file.filename.lower().endswith('.wav'):
            return jsonify({"error": "仅支持WAV格式文件"}), 400
        
        # 保存上传的音频到临时文件
        temp_path = "temp_audio.wav"
        audio_file.save(temp_path)
        
        # 验证WAV文件的格式参数是否符合要求
        try:
            with contextlib.closing(wave.open(temp_path, 'r')) as wav_file:
                n_channels = wav_file.getnchannels()  # 声道数
                sample_width = wav_file.getsampwidth()  # 采样宽度
                frame_rate = wav_file.getframerate()  # 采样率
                
                # 检查音频是否为单声道
                if n_channels != 1:
                    return jsonify({
                        "error": f"仅支持单声道音频，实际为{n_channels}声道"
                    }), 400
                
                # 检查采样率是否为16kHz
                if frame_rate != 16000:
                    return jsonify({
                        "error": f"采样率应为16000Hz，实际为{frame_rate}Hz"
                    }), 400
                
                # 检查位深度是否为16-bit
                if sample_width != 2:
                    return jsonify({
                        "error": f"仅支持16-bit音频，实际为{sample_width*8}-bit"
                    }), 400
        except Exception as e:
            return jsonify({"error": f"无效的WAV文件: {str(e)}"}), 400
        
        # 加载音频数据并调整采样率
        y, sr = librosa.load(temp_path, sr=16000, mono=True)
        
        # 提取音频特征
        input_features = processor.feature_extractor(y, sampling_rate=16000, return_tensors="np").input_features
        
        # 转换为CTranslate2需要的格式
        features = ctranslate2.StorageView.from_array(input_features)
        
        # 构建Whisper模型需要的特殊标记序列
        tokenizer = processor.tokenizer
        sot_token = tokenizer.convert_tokens_to_ids("<|startoftranscript|>")  # 转录开始标记

        # 根据模式动态设置任务和语言标记
        if mode == 'english':
            # 英文模式：将所有语言翻译成英文
            language_token = tokenizer.convert_tokens_to_ids("<|en|>")      # 英文语言标记
            task_token = tokenizer.convert_tokens_to_ids("<|translate|>") # 翻译任务标记
        else:
            # 中文模式（默认）：转录中文
            language_token = tokenizer.convert_tokens_to_ids("<|zh|>")      # 中文语言标记
            task_token = tokenizer.convert_tokens_to_ids("<|transcribe|>") # 转录任务标记

        prompt = [[sot_token, language_token, task_token]]  # 组装prompt
        
        # 执行语音识别推理
        results = translator.generate(features, prompts=prompt, beam_size=5, sampling_temperature=0.2)
        
        # 解码结果为文本
        transcription_raw = processor.tokenizer.decode(results[0].sequences_ids[0], skip_special_tokens=True).strip()
        
        # 使用 FunASR 进行标点恢复
        final_text = transcription_raw
        if punc_model is not None and mode == 'chinese' and transcription_raw:
            try:
                # 使用 FunASR 的 ct-punc 模型进行标点恢复
                # 打印原始文本，以便调试
                print(f"原始文本: {transcription_raw}")
                
                # 确保文本不为空
                if transcription_raw.strip():
                    # 调用 FunASR 模型进行标点恢复
                    punc_result = punc_model.generate(input=transcription_raw)
                    print(f"标点恢复结果: {punc_result}")
                    
                    # 正确解析 FunASR 的输出格式
                    if isinstance(punc_result, list) and len(punc_result) > 0:
                        if isinstance(punc_result[0], dict) and "text" in punc_result[0]:
                            final_text = punc_result[0]["text"]
                        else:
                            print(f"警告: 标点恢复结果格式不符合预期: {punc_result}")
            except Exception as e:
                print(f"标点恢复过程中出错: {str(e)}")
                # 如果标点恢复失败，使用原始文本
                final_text = transcription_raw

        return jsonify({"text": final_text})
    except Exception as e:
        import traceback
        print(f"处理音频时发生错误: {str(e)}")
        print(traceback.format_exc())  # 打印完整的错误堆栈
        return jsonify({"error": f"处理音频时发生错误: {str(e)}"}), 500
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

# 启动应用
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True)
