# Whisper语音识别展示

## 下载核心依赖

```bash
pip install flask numpy librosa soundfile
pip install transformers>=4.30.0 ctranslate2>=3.17.0
pip install -U funasr
pip install modelscope
pip install requests
```

## 相关模型

- merged_model 微调后的模型与基础模型合并
- ctranslate_model CTranslate2 格式的 Whisper 模型
- FunASR 标点恢复模型
- 在阿里云申请模型 api
- 在火山引擎申请豆包语音合成 api

## 目录结构
```
/mnt/d/WSL/asr_large_model/web/
├── app.py                  
├── merged_model/           
├── ctranslate_model/      
├── static/                 
    └── script.js
└── templates/            
    └── index.html        
```

## 运行代码
```bash
python app.py
```
打开静态网站