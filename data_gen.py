import pandas as pd
import os
import argparse

# 🔧 argparse 설정
parser = argparse.ArgumentParser(description="TXT 파일을 읽어 이미지/텍스트 경로를 CSV로 변환")
parser.add_argument(
    "--base_dir",
    type=str,
    default="/workspace/data/findit2",
    help="작업할 기본 디렉토리 경로"
)
args = parser.parse_args()

# 📂 작업 경로
base_dir = args.base_dir

# 📄 처리할 txt 파일 목록 및 하위 폴더명
txt_files = [
    ("train.txt", "train"),
    ("val.txt", "val"),
    ("test.txt", "test"),
]

for txt_file, subdir in txt_files:
    txt_path = os.path.join(base_dir, txt_file)
    image_dir = os.path.join(base_dir, subdir)  # 이미지/텍스트 파일이 있는 디렉토리

    if not os.path.exists(txt_path):
        print(f"⚠️ {txt_path} 가 존재하지 않습니다. 건너뜁니다.")
        continue

    # txt 파일 읽어서 DataFrame으로 파싱
    df = pd.read_csv(
        txt_path,
        header=0,
        quotechar='"',
        skipinitialspace=True
    )

    # image 컬럼 체크
    if "image" not in df.columns:
        print(f"⚠️ {txt_file} 에 'image' 컬럼이 없습니다. ocr 컬럼을 생성하지 않습니다.")
        continue

    # 🔧 새로운 데이터 저장 리스트
    valid_rows = []

    for _, row in df.iterrows():
        image_name = row["image"]
        if not isinstance(image_name, str):
            continue

        image_path = os.path.join(image_dir, image_name)
        ocr_name = image_name.replace(".png", ".txt")
        ocr_path = os.path.join(image_dir, ocr_name)

        # 실제 파일 존재 여부 확인
        if os.path.exists(image_path) and os.path.exists(ocr_path):
            new_row = row.copy()
            new_row["image"] = image_path
            new_row["ocr"] = ocr_path
            valid_rows.append(new_row)

    # DataFrame 생성
    filtered_df = pd.DataFrame(valid_rows)

    # 동일한 이름의 csv 경로
    csv_name = os.path.splitext(txt_file)[0] + ".csv"
    csv_path = os.path.join(base_dir, csv_name)

    # CSV 저장
    filtered_df.to_csv(csv_path, index=False)
    print(f"✅ {csv_path} 저장 완료! (총 {len(filtered_df)}행, 실제 파일 존재 확인)")
