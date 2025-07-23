# 📌 ReceiptTamperOCR

## 🌎 환경
- GPU: **RTX 4070 Ti SUPER**
- CPU: **AMD Ryzen 5 7500F 6-Core Processor**
- CUDA: **11.8**
- cuDNN: **11.8**
- 🛠️ 나머지 파이썬 패키지는 아래 명령으로 설치:
pip install -r requirements.txt

---

## 📌 데이터 생성 & 학습 순서

### 1️⃣ OSTF 데이터셋 다운로드
https://drive.google.com/file/d/16Pyv7nLBOsOefwzdCsa0ndXxnzknfxtw/view

### 2️⃣ OSTF 데이터셋 압축 해제
unzip OSTF.zip  
비밀번호: OSTF-INTSIG-DLVC-411

### 3️⃣ 데이터셋 CSV 생성
python3 tools/data_gen.py  
(base_dir, output 경로는 config.yaml에 맞게 설정)

### 4️⃣ 학습 실행
python3 train.py

---

## ⚙️ config.yaml 설명
```yaml
# 학습할 모델 선택: image 또는 text
model_type: text  # or "image"

# 공통 하이퍼파라미터
seed: 42
epochs: 10
batch_size: 32
num_workers: 4
patience: 3

# 이미지 모델 하이퍼파라미터
image_lr: 1e-5

# 텍스트 모델 하이퍼파라미터
text_lr: 2e-5
max_length: 512

# 데이터 경로
base_dir: /workspace/ReceiptTamperOCR/OSTF
train_csv: output/train.csv
test_csv: output/test.csv
