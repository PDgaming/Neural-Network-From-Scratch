python cli.py train \
  --dataset iris \
  --architecture architecture.json \
  --loss categorical_crossentropy \
  --optimizer adam \
  --lr 0.01 \
  --epochs 1000 \
  --save models/my_model