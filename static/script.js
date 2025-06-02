// 获取页面中的 DOM 元素
const recordBtn = document.getElementById('recordBtn');
const resultDiv = document.getElementById('result');
const audioPlayer = document.getElementById('audio-player');
const statusDiv = document.getElementById('status');

let mediaRecorder;
// 用于存储录音数据
let audioChunks = []; 
// 音频上下文（Web Audio API 对象）
let audioContext = null;

// 将音频数据编码为 WAV 格式（16kHz、16bit PCM）
function encodeWAV(samples) {
    const buffer = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(buffer);
    
    // WAV头部
    const writeString = (view, offset, string) => {
        for (let i = 0; i < string.length; i++) {
            view.setUint8(offset + i, string.charCodeAt(i));
        }
    };
    
    const sampleRate = 16000;
    const numChannels = 1;
    const bytePerSample = 2;
    const blockAlign = numChannels * bytePerSample;
    
    writeString(view, 0, 'RIFF');
    view.setUint32(4, 36 + samples.length * 2, true);
    writeString(view, 8, 'WAVE');
    writeString(view, 12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true); // PCM格式
    view.setUint16(22, numChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * blockAlign, true); // 字节率
    view.setUint16(32, blockAlign, true);
    view.setUint16(34, bytePerSample * 8, true); // 位深
    writeString(view, 36, 'data');
    view.setUint32(40, samples.length * 2, true);
    
    // 写入PCM数据
    const floatTo16BitPCM = (output, offset, input) => {
        for (let i = 0; i < input.length; i++, offset += 2) {
            const s = Math.max(-1, Math.min(1, input[i]));
            output.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
        }
    };
    
    floatTo16BitPCM(view, 44, samples);
    return new Blob([view], { type: 'audio/wav' });
}

// 降采样函数（非16kHz时转换为16kHz）
function downsampleBuffer(buffer, sampleRate, targetSampleRate) {
    if (targetSampleRate === sampleRate) {
        return buffer;
    }
    
    const ratio = sampleRate / targetSampleRate;
    const newLength = Math.round(buffer.length / ratio);
    const result = new Float32Array(newLength);
    let offsetResult = 0;
    let offsetBuffer = 0;
    
    while (offsetResult < newLength) {
        const nextOffsetBuffer = Math.round((offsetResult + 1) * ratio);
        let accum = 0;
        let count = 0;
        
        for (let i = offsetBuffer; i < nextOffsetBuffer && i < buffer.length; i++) {
            accum += buffer[i];
            count++;
        }
        
        result[offsetResult] = accum / count;
        offsetResult++;
        offsetBuffer = nextOffsetBuffer;
    }
    
    return result;
}

// 创建新的音频上下文
function createAudioContext() {
    if (!audioContext) {
        audioContext = new (window.AudioContext || window.webkitAudioContext)({
            sampleRate: 16000
        });
    }
    return audioContext;
}

// 主流程：点击按钮开始或停止录音
recordBtn.addEventListener('click', async () => {
    if (!mediaRecorder) {
        try {
            statusDiv.textContent = "正在初始化录音设备...";
            
            // 获取用户媒体流
            const stream = await navigator.mediaDevices.getUserMedia({ 
                audio: {
                    sampleRate: 16000,
                    channelCount: 1,
                    echoCancellation: true,
                    noiseSuppression: true
                }
            });
            
            // 创建MediaRecorder
            mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm;codecs=opus' });
            
            // 收集音频数据
            audioChunks = [];
            mediaRecorder.ondataavailable = event => {
                audioChunks.push(event.data);
            };
            
            mediaRecorder.onstop = async () => {
                statusDiv.textContent = "正在处理音频...";
                
                try {
                    // 将录音数据合并为Blob
                    const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                    
                    // 创建新的音频上下文用于解码
                    const decodeContext = createAudioContext();
                    
                    // 将Blob转换为ArrayBuffer
                    const arrayBuffer = await audioBlob.arrayBuffer();
                    
                    // 解码音频数据
                    const audioData = await decodeContext.decodeAudioData(arrayBuffer);
                    
                    // 获取原始PCM数据
                    const originalSamples = audioData.getChannelData(0);
                    
                    // 如果实际采样率不是16kHz，进行降采样
                    let finalSamples = originalSamples;
                    if (Math.abs(audioData.sampleRate - 16000) > 100) {
                        finalSamples = downsampleBuffer(originalSamples, audioData.sampleRate, 16000);
                    }
                    
                    // 创建WAV格式的Blob
                    const wavBlob = encodeWAV(finalSamples);
                    audioPlayer.src = URL.createObjectURL(wavBlob);
                    
                    // 发送到后端识别
                    statusDiv.textContent = "正在识别语音...";
                    resultDiv.textContent = "识别中...";
                    
                    // 调用模型接口
                    const formData = new FormData();
                    formData.append('audio', wavBlob, 'recording.wav');
                    
                    const response = await fetch('/transcribe', {
                        method: 'POST',
                        body: formData
                    });
                    
                    if (!response.ok) {
                        throw new Error(`服务器错误: ${response.status}`);
                    }
                    
                    const data = await response.json();
                    if (data.error) {
                        resultDiv.innerHTML = `<span class="error">错误: ${data.error}</span>`;
                        if (data.actual) {
                            resultDiv.innerHTML += `<br>实际采样率: ${data.actual}Hz，期望: 16000Hz`;
                        }
                    } else {
                        resultDiv.textContent = data.text;
                    }
                    
                    statusDiv.textContent = "识别完成";
                } catch (e) {
                    console.error('音频处理错误:', e);
                    resultDiv.innerHTML = `<span class="error">处理错误: ${e.message}</span>`;
                    statusDiv.textContent = "处理失败";
                } finally {
                    // 清理资源
                    if (audioContext) {
                        audioContext.close();
                        audioContext = null;
                    }
                }
            };
            
            // 开始录音
            mediaRecorder.start();
            recordBtn.textContent = '停止录音';
            statusDiv.textContent = "录音中...";
        } catch (err) {
            console.error('录音初始化失败:', err);
            statusDiv.innerHTML = `<span class="error">麦克风访问失败: ${err.message}</span>`;
            
            // 清理可能的部分初始化
            if (mediaRecorder) {
                mediaRecorder.stream.getTracks().forEach(track => track.stop());
                mediaRecorder = null;
            }
            if (audioContext) {
                audioContext.close();
                audioContext = null;
            }
        }
    } else {
        // 停止录音
        mediaRecorder.stop();
        mediaRecorder.stream.getTracks().forEach(track => track.stop());
        mediaRecorder = null;
        recordBtn.textContent = '开始录音';
        statusDiv.textContent = "正在处理录音...";
    }
});