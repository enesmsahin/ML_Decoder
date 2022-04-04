import json

label_tsv_list = (
    "/home/deepuser/deepnas/DISK4/DISK4/Enes/data/oi/model_0060000/0/labels/label.tsv",
    "/home/deepuser/deepnas/DISK4/DISK4/Enes/data/oi/model_0060000/1/labels/label.tsv",
    "/home/deepuser/deepnas/DISK4/DISK4/Enes/data/oi/model_0060000/2/labels/label.tsv",
    "/home/deepuser/deepnas/DISK4/DISK4/Enes/data/oi/model_0060000/3/labels/label.tsv",
    "/home/deepuser/deepnas/DISK4/DISK4/Enes/data/oi/model_0060000/4/labels/label.tsv",
    "/home/deepuser/deepnas/DISK4/DISK4/Enes/data/oi/model_0060000/5/labels/label.tsv",
    "/home/deepuser/deepnas/DISK4/DISK4/Enes/data/oi/model_0060000/6/labels/label.tsv",
    "/home/deepuser/deepnas/DISK4/DISK4/Enes/data/oi/model_0060000/7/labels/label.tsv",
)

BLACKLIST = [
    "auto part", "bathroom accessory", "bicycle wheel", "boy", "building", "clothing",
    "door handle", "fashion accessory", "footwear", "girl", "hiking equipment", "human arm",
    "human beard", "human body", "human ear", "human eye", "human face", "human foot",
    "human hair", "human hand", "human head", "human leg", "human mouth", "human nose",
    "land vehicle", "mammal", "man", "person", "personal care", "plant", "plumbing fixture",
    "seat belt", "skull", "sports equipment", "tire", "tree", "vehicle registration plate",
    "wheel", "woman"
]

total_images = 0
num_zero_allowed_class_images = 0
for label_tsv in label_tsv_list:
    with open(label_tsv, "r") as label_tsv_f:
        line_no = 1
        for line in label_tsv_f:
            total_images += 1
            id, preds = line.split("\t")
            preds = json.loads(preds)
            allowed_preds = []
            for pred in preds:
                if pred["class"].lower() not in BLACKLIST:
                    allowed_preds.append(pred)
            if len(allowed_preds) == 0:
                num_zero_allowed_class_images += 1
                print(f"Allowed classes are 0 for {label_tsv}\tLine: {line_no}\tImage ID: {id}")
            line_no += 1

print(f"Check done!\nTotal Images: {total_images}\n \
    Number of Images with 0 allowed classes: {num_zero_allowed_class_images}")