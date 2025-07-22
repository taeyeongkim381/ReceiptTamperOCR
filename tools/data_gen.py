import os
import cv2
import pandas as pd
from tqdm import tqdm
import pickle
from easyocr import Reader
import matplotlib.pyplot as plt
import argparse

# ===================== EasyOCR 초기화 =====================
reader = Reader(['en'], gpu=True)

# ===================== OCR 함수 =====================
def ocr_image_crop_easyocr(image_path, x1, y1, x2, y2, pad):
    img = cv2.imread(image_path)
    if img is None:
        return None
    h, w = img.shape[:2]
    x1p = max(int(x1) - pad, 0)
    y1p = max(int(y1) - pad, 0)
    x2p = min(int(x2) + pad, w)
    y2p = min(int(y2) + pad, h)
    crop = img[y1p:y2p, x1p:x2p]
    if crop.size == 0:
        return None

    results = reader.readtext(crop)
    filtered = [text for _, text, conf in results if conf >= 0.5]
    return "\n".join(filtered).strip() if filtered else None

# ===================== PK -> CSV 저장 =====================
def save_csv_from_pk(pks_dir, mmacc_base, csv_dir, pk_files):
    os.makedirs(csv_dir, exist_ok=True)
    for pk in pk_files:
        pk_path = os.path.join(pks_dir, pk)
        if not os.path.exists(pk_path):
            print(f"missing file: {pk_path}")
            continue
        print(f"processing: {pk}")
        with open(pk_path, "rb") as f:
            data = pickle.load(f)
        rows = []
        for rel_img_path, info in tqdm(data.items(), desc=pk):
            abs_img_path = os.path.join(mmacc_base, os.path.relpath(rel_img_path, "mmacc"))
            for b in info.get("b", []):
                x1, y1, x2, y2, label = b
                rows.append([abs_img_path, x1, y1, x2, y2, label])
        df = pd.DataFrame(rows, columns=["image_path", "x1", "y1", "x2", "y2", "label"])
        out_path = os.path.join(csv_dir, pk.replace(".pk", ".csv"))
        df.to_csv(out_path, index=False)
        print(f"saved: {out_path}")

# ===================== OCR -> TXT =====================
def generate_txt_from_csv(csv_dir, txt_base, pad):
    os.makedirs(txt_base, exist_ok=True)
    csv_files = [f for f in os.listdir(csv_dir) if f.endswith(".csv")]
    for csv_file in csv_files:
        dataset_name = os.path.splitext(csv_file)[0]
        csv_path = os.path.join(csv_dir, csv_file)
        df = pd.read_csv(csv_path)
        save_dir = os.path.join(txt_base, dataset_name)
        os.makedirs(save_dir, exist_ok=True)
        grouped = df.groupby("image_path")
        print(f"OCR running: {csv_file} (images: {len(grouped)})")

        created_txt_count = 0
        total_text_lines = 0

        for image_path, group in tqdm(grouped, desc=dataset_name):
            img_name = os.path.basename(image_path)
            txt_name = os.path.splitext(img_name)[0] + ".txt"
            txt_path = os.path.join(save_dir, txt_name)

            x1_min = group["x1"].min()
            y1_min = group["y1"].min()
            x2_max = group["x2"].max()
            y2_max = group["y2"].max()

            text = ocr_image_crop_easyocr(image_path, x1_min, y1_min, x2_max, y2_max, pad)
            if text and text.strip():
                created_txt_count += 1
                line_count = len([t for t in text.splitlines() if t.strip()])
                total_text_lines += line_count
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(text.strip())

        print(f"OCR done: {dataset_name} | total: {len(grouped)} | success: {created_txt_count} | discarded: {len(grouped)-created_txt_count} | lines: {total_text_lines}")

# ===================== merged CSV 생성 =====================
def create_merged_csv(csv_dir, txt_base, merged_csv_dir):
    os.makedirs(merged_csv_dir, exist_ok=True)
    csv_files = [f for f in os.listdir(csv_dir) if f.endswith(".csv")]
    for csv_file in csv_files:
        dataset_name = os.path.splitext(csv_file)[0]
        csv_path = os.path.join(csv_dir, csv_file)
        txt_dir = os.path.join(txt_base, dataset_name)
        if not os.path.exists(txt_dir):
            print(f"missing txt folder: {txt_dir}")
            continue
        df = pd.read_csv(csv_path)
        grouped = df.groupby("image_path")
        rows = []
        for image_path, group in tqdm(grouped, desc=dataset_name):
            img_name = os.path.basename(image_path)
            txt_name = os.path.splitext(img_name)[0] + ".txt"
            txt_path = os.path.join(txt_dir, txt_name)
            if not os.path.exists(txt_path):
                continue
            with open(txt_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
            if not content:
                continue
            # 핵심 수정: bbox 중 하나라도 1이 있으면 1
            label = 1 if (group["label"] == 1).any() else 0
            rows.append([image_path, label, txt_path])
        merged_df = pd.DataFrame(rows, columns=["image_path", "label", "txt_path"])
        out_path = os.path.join(merged_csv_dir, f"{dataset_name}_merged.csv")
        merged_df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"saved merged: {out_path} ({len(merged_df)} rows)")

# ===================== train/test split =====================
def split_by_train_test(merged_csv_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    train_rows, test_rows = [], []

    merged_files = [f for f in os.listdir(merged_csv_dir) if f.endswith("_merged.csv")]
    for f in merged_files:
        path = os.path.join(merged_csv_dir, f)
        df = pd.read_csv(path)
        if "_train" in f:
            train_rows.append(df)
            print(f"train read: {f} ({len(df)} rows)")
        elif "_test" in f:
            test_rows.append(df)
            print(f"test read: {f} ({len(df)} rows)")

    train_df = pd.concat(train_rows, ignore_index=True) if train_rows else pd.DataFrame(columns=["image_path","label","txt_path"])
    test_df  = pd.concat(test_rows, ignore_index=True) if test_rows else pd.DataFrame(columns=["image_path","label","txt_path"])

    train_path = os.path.join(output_dir, "train.csv")
    test_path = os.path.join(output_dir, "test.csv")
    train_df.to_csv(train_path, index=False, encoding="utf-8-sig")
    test_df.to_csv(test_path, index=False, encoding="utf-8-sig")
    print(f"saved train.csv: {len(train_df)} rows")
    print(f"saved test.csv: {len(test_df)} rows")

    # 간단한 레이블 분포 확인
    def save_plot(df, title, filename):
        if df.empty:
            return
        counts = df["label"].value_counts()
        plt.figure()
        counts.plot(kind="bar")
        plt.title(title)
        plt.xlabel("Label")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()

    save_plot(train_df, "Train Label Distribution", "train_label_distribution.png")
    save_plot(test_df, "Test Label Distribution", "test_label_distribution.png")

# ===================== 메인 =====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pks_dir", default="/workspace/ReceiptTamperOCR/OSTF/pks")
    parser.add_argument("--mmacc_base", default="/workspace/ReceiptTamperOCR/OSTF/mmacc")
    parser.add_argument("--csv_dir", default="/workspace/ReceiptTamperOCR/OSTF/csv")
    parser.add_argument("--txt_base", default="/workspace/ReceiptTamperOCR/OSTF/txt")
    parser.add_argument("--merged_csv_dir", default="/workspace/ReceiptTamperOCR/OSTF/merged_csv")
    parser.add_argument("--output_dir", default="/workspace/ReceiptTamperOCR/OSTF/output")
    parser.add_argument("--pad", type=int, default=5)
    args = parser.parse_args()

    for path in [args.csv_dir, args.txt_base, args.merged_csv_dir, args.output_dir]:
        os.makedirs(path, exist_ok=True)

    pk_files = [
        "anytext_test.pk", "anytext_train.pk",
        "derend_test.pk", "derend_train.pk",
        "diffste_test.pk", "diffste_train.pk",
        "mostel_test.pk", "mostel_train.pk",
        "srnet_test.pk", "srnet_train.pk",
        "stefann_test.pk", "stefann_train.pk",
        "textdiff_test.pk", "textdiff_train.pk",
        "textocr_test.pk", "textocr_train.pk",
        "udifftext_test.pk", "udifftext_train.pk",
    ]

    save_csv_from_pk(args.pks_dir, args.mmacc_base, args.csv_dir, pk_files)
    generate_txt_from_csv(args.csv_dir, args.txt_base, args.pad)
    create_merged_csv(args.csv_dir, args.txt_base, args.merged_csv_dir)
    split_by_train_test(args.merged_csv_dir, args.output_dir)
