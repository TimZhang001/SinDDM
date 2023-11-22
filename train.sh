python main.py --scope mvtec_hazelnut --mode train --augment 1 --device_num 5 --load_milestone 40 --dim 160 --train_num_steps 120001

python main.py --scope mvtec_hazelnut_240 --mode train --augment 1 --device_num 6 --load_milestone 0 --dim 320 --train_num_steps 60001

python main.py --scope mvtec_carpet_aug --mode train --dataset_folder ./datasets/mvtec/carpet/ --image_name metal_contamination_005.png --augment 1 --device_num 7 --load_milestone 0 --train_num_steps 120001


python main.py --scope mvtec_hazelnut --mode sample --augment 1 --device_num 6 --load_milestone 12