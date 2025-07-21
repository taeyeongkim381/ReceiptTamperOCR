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
### 1. 데이터 다운로드
```bash
python3 download_findit.py
```

### 2. LMDB → .mdb에서 image.jpg와 mask.png 확보
```bash
python3 data_gen.py
```

### 3. 오버샘플링된 train 데이터 생성
```bash
python3 train_oversampled.py
```

### 4. 학습
```bash
python3 train.py
```