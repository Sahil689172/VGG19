import os
import shutil
import random

base_path = os.getcwd()

classes = ["Benign", "Malignant", "Normal"]

train_dir = os.path.join(base_path, "dataset", "train")
test_dir = os.path.join(base_path, "dataset", "test")

valid_ext = (".jpg", ".jpeg", ".png", ".bmp")

for cls in classes:
    # 🔥 FIXED PATH (IMPORTANT)
    src_folder = os.path.join(base_path, cls, cls, "image")

    if not os.path.exists(src_folder):
        print(f"❌ Path not found: {src_folder}")
        continue

    os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
    os.makedirs(os.path.join(test_dir, cls), exist_ok=True)

    images = [img for img in os.listdir(src_folder)
              if img.lower().endswith(valid_ext)]

    if len(images) == 0:
        print(f"⚠️ No images found in {cls}")
        continue

    random.shuffle(images)

    split_idx = int(0.8 * len(images))

    train_imgs = images[:split_idx]
    test_imgs = images[split_idx:]

    for img in train_imgs:
        shutil.copy2(os.path.join(src_folder, img),
                     os.path.join(train_dir, cls, img))

    for img in test_imgs:
        shutil.copy2(os.path.join(src_folder, img),
                     os.path.join(test_dir, cls, img))

    print(f"✅ {cls}: {len(train_imgs)} train | {len(test_imgs)} test")

print("\n🎉 Dataset split COMPLETED successfully!")