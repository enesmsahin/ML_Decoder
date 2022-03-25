echo "Generating Test"
python save_results_coco.py --img_id_file=/media/mmrg/DATA/enes/data/COCO/coco_caption/test.label.tsv --out-file=coco_test_ml_decoder.label.tsv
echo "Generating Val"
python save_results_coco.py --img_id_file=/media/mmrg/DATA/enes/data/COCO/coco_caption/val.label.tsv --out-file=coco_val_ml_decoder.label.tsv
echo "Generating Train"
python save_results_coco.py --img_id_file=/media/mmrg/DATA/enes/data/COCO/coco_caption/train.label.tsv --out-file=coco_train_ml_decoder.label.tsv
echo "Done!"