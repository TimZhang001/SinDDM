python train_sample.py --scope all_sample --mode train --dataset_folder ./datasets/mvtec/leather/glue/ --save_sample_every 1000 --device_num 5
python train_sample.py --scope all_sample --mode sample --dataset_folder ./datasets/mvtec/grid/bent/ --save_sample_every 1000 --device_num 6 --load_milestone 5

datasets/mvtec/leather/glue