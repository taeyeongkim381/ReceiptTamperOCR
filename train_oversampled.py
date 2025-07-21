import pandas as pd
import os
import argparse

# 🔧 argparse 설정
parser = argparse.ArgumentParser(description="train.csv의 클래스 불균형을 오버샘플링으로 보정")
parser.add_argument(
    "--base_dir",
    type=str,
    default="/workspace/data/findit2",
    help="train.csv가 위치한 기본 디렉토리 경로"
)
args = parser.parse_args()

# 📂 경로 설정
base_dir = args.base_dir
train_csv = os.path.join(base_dir, "train.csv")

# 📄 CSV 읽기
df = pd.read_csv(train_csv)
df_pos = df[df["forged"] == 1]   # 조작된 것
df_neg = df[df["forged"] == 0]   # 조작 안 된 것

pos_count = len(df_pos)
neg_count = len(df_neg)

print(f"원본 | forged=1: {pos_count}, forged=0: {neg_count}")

# 📈 오버샘플링
if pos_count < neg_count:
    # 조작된 것(forged=1)을 복제
    factor = neg_count // pos_count
    remainder = neg_count % pos_count
    df_pos_oversampled = pd.concat([df_pos] * factor, ignore_index=True)
    if remainder > 0:
        df_pos_oversampled = pd.concat(
            [df_pos_oversampled, df_pos.sample(remainder, replace=True)],
            ignore_index=True
        )
    balanced_df = pd.concat([df_neg, df_pos_oversampled], ignore_index=True)
else:
    # 조작 안 된 것(forged=0)을 복제
    factor = pos_count // neg_count
    remainder = pos_count % neg_count
    df_neg_oversampled = pd.concat([df_neg] * factor, ignore_index=True)
    if remainder > 0:
        df_neg_oversampled = pd.concat(
            [df_neg_oversampled, df_neg.sample(remainder, replace=True)],
            ignore_index=True
        )
    balanced_df = pd.concat([df_pos, df_neg_oversampled], ignore_index=True)

# 🔀 데이터 섞기
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

print("오버샘플링 후 클래스 분포:")
print(balanced_df["forged"].value_counts())

# 💾 새로운 CSV 저장
balanced_csv = os.path.join(base_dir, "train_balanced.csv")
balanced_df.to_csv(balanced_csv, index=False)
print(f"✅ Balanced CSV 저장 완료: {balanced_csv}")
