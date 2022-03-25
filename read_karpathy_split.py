import json

json_path = "/media/mmrg/DATA/enes/data/COCO/caption_datasets/dataset_coco.json"

with open(json_path, "r") as f:
    json_data = json.load(f)

train_out = "coco_train_set.txt"
val_out = "coco_val_set.txt"
test_out = "coco_test_set.txt"
restval_out = "coco_restval_set.txt"

with open(train_out, "w") as train_f, open(val_out, "w") as val_f, open(test_out, "w") as test_f, open(restval_out, "w") as restval_f:
    for im_dict in json_data["images"]:
        split = im_dict["split"]
        if split == "train":
            f = train_f
        elif split == "val":
            f = val_f
        elif split == "test":
            f = test_f
        elif split == "restval":
            f = restval_f
        else:
            raise Exception(f"Unknown split type: {split}")
        f.write(str(im_dict["filename"]) + "\n")

print("Done!")