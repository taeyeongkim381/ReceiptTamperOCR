# 🌎 환경
- GPU: **RTX 4070 Ti SUPER**
- CPU: **AMD Ryzen 5 7500F 6-Core Processor**
- CUDA: **11.8**
- cuDNN: **11.8**
- 🛠️ 나머지 파이썬 패키지는 아래 명령으로 설치:
```bash
pip install -r requirements.txt
```

# 📌 데이터 생성 & 학습 순서
### 1. OSTF 데이터셋 .zip 파일 다운로드
```bash
https://drive.google.com/file/d/16Pyv7nLBOsOefwzdCsa0ndXxnzknfxtw/view
```

### 2. OSTF 데이터셋 압축 해제
```bash
unzip OSTF.zip (비밀번호: OSTF-INTSIG-DLVC-411)
```

### 3. 데이터셋 생성 (절대 경로로 작성)
```bash
python3 tools/data_gen.py
```

### 4. 학습
```bash
python3 train.py
```