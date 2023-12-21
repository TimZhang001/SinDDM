#!/bin/bash

# 定义数据集中的所有类型
categories=('carpet' 'grid' 'leather' 'tile' 'wood' 'bottle' 'cable' 'capsule'  'metal_nut' ,'hazelnut','pill' 'screw' 'toothbrush' 'transistor' 'zipper')
#categories=('')

# 循环处理每个类型
for category in "${categories[@]}"; do
    
    base_paths="/home/zhangss/PHDPaper/04_SinDDM/results/${category}"

    # 寻找source_path包含的所有子目录
    sub_dirs=$(ls "${base_paths}")

    # 循环处理每个子目录
    for sub_dir in ${sub_dirs}; do

        # print sub_dir
        echo "${category}/${sub_dir}"
        
        # 真实样本的路径
        src_paths = "/home/zhangss/PHDPaper/04_SinDDM/datasets/mvtec/${category}/${sub_dir}/tmp/scale_4/000.png"

        # 设置其他路径
        root_paths="/home/zhangss/PHDPaper/04_SinDDM/results/${category}/${sub_dir}/single_image" 
        real_paths="/home/zhangss/PHDPaper/04_SinDDM/results/${category}/${sub_dir}/single_image/sample_true" 
        fake_paths="/home/zhangss/PHDPaper/04_SinDDM/results/${category}/${sub_dir}/single_image/sample_eval"

        # 如果real_paths不存在，先创建
        if [ ! -d "${real_paths}" ]; then
            mkdir -p "${real_paths}"
        fi

        # 将真实样本复制到real_paths
        cp "${src_paths}" "${real_paths}"

        # 调用处理脚本进行训练
        python third_party/Metric/oneshot_evaluate.py --root_path "${root_paths}" --real_path "${real_paths}" --fake_path "${fake_paths}"

    echo "----------------------------------------------"

done