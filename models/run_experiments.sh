#!/bin/bash
# ===============================
# 批量运行多组实验的脚本
# 适用于 MultiInputModelWithAttention 五分类任务
# ===============================

# 可自定义变量
PROJECTS=("Eclipse")   # 项目名称列表 "GCC" "Eclipse" "Mozilla"
SEEDS=(42)                    # 随机种子 42
EPOCHS=(5)                           # 训练 epoch 数
# 每组权重用引号括起来（空格隔开）
GAMMA=(1)
CLASS_WEIGHTS_LIST=(
  "1 1 1 1 1"
)

# Python 脚本路径（根据你的代码存放位置修改）
SCRIPT_PATH="./CAF-Net.py"

# 输出根目录
RESULTS_DIR="./results"


# ===============================
# 循环执行实验
# ===============================
for project in "${PROJECTS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    for epoch in "${EPOCHS[@]}"; do
      for gamma in "${GAMMA[@]}"; do
          for weights in "${CLASS_WEIGHTS_LIST[@]}"; do

            # 生成可读性强的权重字符串
            WEIGHT_TAG=$(echo "$weights" | tr ' ' '-')

            # 构造输出目录
            OUT_DIR="${RESULTS_DIR}/${project}/ep${epoch}bs16/seed${seed}/gamma${gamma}/weights-${WEIGHT_TAG}-4090D"

            # 若目录存在，则跳过
            if [ -d "$OUT_DIR" ]; then
              echo "Folder Exists: $OUT_DIR"
              continue
            fi

            echo "Train Begin: Project=${project}, Seed=${seed}, Epoch=${epoch}, Gamma=${gamma}, Weights=[${weights}]"
            mkdir -p "$OUT_DIR"

            # 运行训练脚本
            python "$SCRIPT_PATH" \
              --project "$project" \
              --output_dir "$OUT_DIR" \
              --num_train_epochs "$epoch" \
              --seed "$seed" \
              --gamma $gamma \
              --class_weights $weights \
              > "${OUT_DIR}/train.log" 2>&1

            # 检查是否执行成功
            if [ $? -eq 0 ]; then
              echo "Experiment Finished: $OUT_DIR"
            else
              echo "Experiment Failed: $OUT_DIR"
            fi
          done
      done
    done
  done
done

echo "Experiments Done！"
