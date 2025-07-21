# 📌 데이터 생성 & 학습 순서

```bash
# 1. 데이터 다운로드
python3 download_findit.py

# 2. LMDB → .mdb에서 image.jpg와 mask.png 확보
python3 data_gen.py

# 3. 오버샘플링된 train 데이터 생성
python3 train_oversampled.py
