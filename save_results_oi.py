import json
import os
import os.path as op
import argparse
import time

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data.distributed

from src_files.helper_functions.bn_fusion import fuse_bn_recursively
from src_files.models import create_model
import matplotlib

from src_files.models.tresnet.tresnet import InplacABN_to_ABN

matplotlib.use('TkAgg')
matplotlib.use('TkAgg')
from PIL import Image
import numpy as np
import glob
from pathlib import Path
from tqdm import tqdm

parser = argparse.ArgumentParser(description='PyTorch MS_COCO infer')
parser.add_argument('--num-classes', default=9605, type=int)
parser.add_argument('--model-path', type=str, default='downloadedModels/tresnet_m_open_images_200_groups_86_8.pth')
parser.add_argument('--model-name', type=str, default='tresnet_m')
parser.add_argument('--image-size', type=int, default=224)
# parser.add_argument('--dataset-type', type=str, default='MS-COCO')
parser.add_argument('--th', type=float, default=0.20)
parser.add_argument('--top-k', type=float, default=20)
# ML-Decoder
parser.add_argument('--use-ml-decoder', default=1, type=int)
parser.add_argument('--num-of-groups', default=200, type=int)  # full-decoding
parser.add_argument('--decoder-embedding', default=768, type=int)
parser.add_argument('--zsl', default=0, type=int)
parser.add_argument('--allowed_classes_file', type=str, default="open_images_class-descriptions-boxable_v5.csv", help="Open Images CSV File indicating boxable classes")
parser.add_argument('--trainable_classes_file', type=str, default="oidv6-classes-trainable.txt")
parser.add_argument('--all_classes_file', type=str, default="oidv6-class-descriptions.csv")

prediction_tsv_files = (
    # "/home/deepuser/deepnas/DISK4/DISK4/Enes/data/oi/model_0060000/0/predictions.tsv",
    # "/home/deepuser/deepnas/DISK4/DISK4/Enes/data/oi/model_0060000/1/predictions.tsv",
    # "/home/deepuser/deepnas/DISK4/DISK4/Enes/data/oi/model_0060000/2/predictions.tsv",
    # "/home/deepuser/deepnas/DISK4/DISK4/Enes/data/oi/model_0060000/3/predictions.tsv",
    "/home/deepuser/deepnas/DISK4/DISK4/Enes/data/oi/model_0060000/4/predictions.tsv",
    "/home/deepuser/deepnas/DISK4/DISK4/Enes/data/oi/model_0060000/5/predictions.tsv",
    "/home/deepuser/deepnas/DISK4/DISK4/Enes/data/oi/model_0060000/6/predictions.tsv",
    "/home/deepuser/deepnas/DISK4/DISK4/Enes/data/oi/model_0060000/7/predictions.tsv",
)
img_root_dir = "/home/deepuser/deepnas/DISK4/DISK4/Enes/IMAGES/open_images/train/"

def get_allowed_image_tags_for_open_images(image_tag_file_path: str) -> list:
        """Read allowed image tags to a list from the csv file in the format of
        Open Images dataset.

        Args:
            image_tag_file_path (str): path to csv file containing allowed image tags

        Returns:
            list[str]: list of allowed tags
        """
        tags_list = []
        tag_ids_list = []
        with open(image_tag_file_path) as f:
            for line in f:
                tag_id, tag_name = line.split(",", maxsplit=1)
                tag_name = tag_name.strip()
                # tag_name = tag_name.split("(")[0].strip() # Remove tags with explanatory parentheses
                tags_list.append(tag_name)
                tag_ids_list.append(tag_id)

        return tags_list, tag_ids_list

def get_orig_trainable_class_names(trainable_class_file_path, all_classes_file_path, out_file=None):
    trainable_cls_ids = []
    with open(trainable_class_file_path, "r") as trainable_f:
        for line in trainable_f:
            cls_id = line.strip()
            trainable_cls_ids.append(cls_id)

    trainable_cls_names = []
    ordered_trainable_cls_ids = []
    
    if out_file is not None:
        with open(out_file, "w") as out_f:
            with open(all_classes_file_path, "r") as all_f:
                for line_all in all_f:
                    cls_id, cls_name = line_all.split(",", maxsplit=1)
                    if cls_id in trainable_cls_ids:
                        out_f.write(line_all)
                        trainable_cls_names.append(cls_name.strip().strip("\""))
                        ordered_trainable_cls_ids.append(cls_id)

    else:
        with open(all_classes_file_path, "r") as all_f:
            for line_all in all_f:
                cls_id, cls_name = line_all.split(",", maxsplit=1)
                if cls_id in trainable_cls_ids:
                    trainable_cls_names.append(cls_name.strip().strip("\""))
                    ordered_trainable_cls_ids.append(cls_id)

    return trainable_cls_names, ordered_trainable_cls_ids

    
def save_allowed_classes_list(save_path, class_list):
    with open(save_path, "w") as f:
        for cls in class_list:
            f.write(cls + "\n")

def main():
    print('Inference code on a single image')

    # parsing args
    args = parser.parse_args()

    # Setup model
    print('creating model {}...'.format(args.model_name))
    model = create_model(args, load_head=True).cuda()
    state = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(state['model'], strict=True)
    ########### eliminate BN for faster inference ###########
    model = model.cpu()
    model = InplacABN_to_ABN(model)
    model = fuse_bn_recursively(model)
    model = model.cuda().half().eval()
    #######################################################
    print('done')


    classes_list = np.array(list(state['idx_to_class'].values()))
    print('done\n')

    classes_list_orig, class_ids_list_orig = get_orig_trainable_class_names(args.trainable_classes_file, args.all_classes_file)
    allowed_image_tags, allowed_image_tag_ids = get_allowed_image_tags_for_open_images(args.allowed_classes_file)

    allowed_indices_in_classes_list = [class_ids_list_orig.index(allowed_cls_id) for allowed_cls_id in allowed_image_tag_ids if allowed_cls_id in class_ids_list_orig]
    classes_list_allowed = classes_list[allowed_indices_in_classes_list]

    # save_allowed_classes_list("allowed_classes_523.txt", classes_list_allowed)

    classes_list_allowed = np.array(classes_list_allowed)
    # doing inference
    print('loading image and doing inference...')
    for prediction_tsv_file in tqdm(prediction_tsv_files):
        print(f"Starting {prediction_tsv_file}")
        with open(prediction_tsv_file, "r") as pred_tsv_f:
            out_path = op.dirname(prediction_tsv_file)
            out_path = op.join(out_path, "labels")
            os.makedirs(out_path)
            with open(op.join(out_path, "label.tsv"), "w") as out_label_tsv_f:
                out_line = ""
                lines = pred_tsv_f.readlines()
                for line in tqdm(lines):
                    img_id, curr_preds = line.split("\t")
                    img_path = op.join(img_root_dir, img_id + ".jpg")
                    try:
                        im = Image.open(img_path)
                    except FileNotFoundError:
                        raise Exception(f"Image Not Found: {img_path}")

                    if im.mode == 'CMYK':
                        im = im.convert('RGB')
                        
                    im_resize = im.resize((args.image_size, args.image_size))

                    np_img = np.array(im_resize, dtype=np.uint8)
                    if im_resize.mode == "L":
                        np_img = np.repeat(np_img[..., np.newaxis], 3, -1)
                    
                    if (np_img.shape[-1] != 3):
                        raise Exception(f"Img is not 3-channel: {img_path}. {np_img.shape}")

                    tensor_img = torch.from_numpy(np_img).permute(2, 0, 1).float() / 255.0  # HWC to CHW
                    tensor_batch = torch.unsqueeze(tensor_img, 0).cuda().half() # float16 inference
                    output = torch.squeeze(torch.sigmoid(model(tensor_batch)))
                    np_output = output.cpu().detach().numpy()

                    ## Top-k predictions
                    # detected_classes = classes_list[np_output > args.th]
                    # idx_sort = np.argsort(-np_output)
                    # detected_classes = np.array(classes_list)[idx_sort][: args.top_k]
                    # scores = np_output[idx_sort][: args.top_k]
                    # idx_th = scores > args.th
                    # detected_classes = detected_classes[idx_th]
                    
                    # im.show()
                    np_output_allowed = np_output[allowed_indices_in_classes_list]
                    idx_sort = np.argsort(-np_output_allowed)
                    detected_classes = classes_list_allowed[idx_sort]
                    scores = np_output_allowed[idx_sort]
                    idx_th = scores > args.th
                    final_detected_classes = detected_classes[idx_th]
                    final_detected_scores = scores[idx_th]

                    if len(final_detected_classes) == 0:
                        print("*" * 10)
                        print(f"Detected classes is zero for {img_path}.")
                        print(f"Max score: {scores.max()}")
                        final_detected_classes = [detected_classes[np.argmax(scores)]]
                        print(final_detected_classes)

                    # line = img_id + "\t" + "["
                    # for detected_class, score in zip(final_detected_classes, final_detected_scores):
                    #     line += f"{{\"class\":\"{detected_class}\",\"conf\":{score}}},"
                    # line = line.rstrip(",") + "]\n"
                    # out_file.write(line)
                    out_json = []
                    for detected_class, score in zip(final_detected_classes, final_detected_scores):
                        out_dict = {
                            "class": detected_class,
                            "conf": score.astype(float)
                        }
                        out_json.append(out_dict)

                    out_line += img_id + "\t" + json.dumps(out_json) + "\n"
                
                out_label_tsv_f.write(out_line)

            print(f"Finished {prediction_tsv_file}")


if __name__ == '__main__':
    main()
