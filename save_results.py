import os
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
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import glob

parser = argparse.ArgumentParser(description='PyTorch MS_COCO infer')
parser.add_argument('--num-classes', default=80, type=int)
parser.add_argument('--model-path', type=str, default='./models_local/TRresNet_L_448_86.6.pth')
parser.add_argument('--pic-path', type=str, default='./pics/000000000885.jpg')
parser.add_argument('--model-name', type=str, default='tresnet_l')
parser.add_argument('--image-size', type=int, default=448)
# parser.add_argument('--dataset-type', type=str, default='MS-COCO')
parser.add_argument('--th', type=float, default=0.75)
parser.add_argument('--top-k', type=float, default=20)
# ML-Decoder
parser.add_argument('--use-ml-decoder', default=1, type=int)
parser.add_argument('--num-of-groups', default=-1, type=int)  # full-decoding
parser.add_argument('--decoder-embedding', default=768, type=int)
parser.add_argument('--zsl', default=0, type=int)
parser.add_argument('--out-file', default="ml_decoder_tag_predictions.tsv", type=str)
parser.add_argument('--img_id_file', default="/home/enes/Desktop/VisionAndLanguageExperimental/Dataset/Flickr30k/image_features_vinvl/flickr_vinvl_img_list.tsv", type=str)
parser.add_argument('--allowed_classes_file', type=str, required=True, help="Open Images CSV File indicating boxable classes")
parser.add_argument('--trainable_classes_file', type=str, default="/home/enes/Desktop/VisionAndLanguageExperimental/Dataset/open_images/oidv6-classes-trainable.txt")
parser.add_argument('--all_classes_file', type=str, default="/home/enes/Desktop/VisionAndLanguageExperimental/Dataset/open_images/oidv6-class-descriptions.csv")


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

    # doing inference
    print('loading image and doing inference...')
    with open(args.out_file, "w") as out_file:
        with open(args.img_id_file, "r") as img_id_file:
            for img_id in img_id_file:
                img_id = img_id.strip("\n")
                im_path = os.path.join(args.pic_path, img_id + ".jpg")
                try:
                    im = Image.open(im_path)
                except FileNotFoundError:
                    print(f"Image not found: {im_path}")
                    continue
                im_resize = im.resize((args.image_size, args.image_size))
                np_img = np.array(im_resize, dtype=np.uint8)
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
                detected_classes = np.array(classes_list_allowed)[idx_sort]
                scores = np_output_allowed[idx_sort]
                idx_th = scores > args.th
                final_detected_classes = detected_classes[idx_th]

                if len(final_detected_classes) == 0:
                    print("*" * 10)
                    print(f"Detected classes is zero for {im_path}.")
                    print(f"Max score: {scores.max()}")
                    final_detected_classes = [detected_classes[np.argmax(scores)]]
                    print(final_detected_classes)

                line = img_id + "\t" + "["
                line += ",".join(final_detected_classes)
                line += "]\n"
                out_file.write(line)


if __name__ == '__main__':
    main()
