python cli.py train \
  --dataset mnist_train \
  --architecture architecture.json \
  --loss categorical_crossentropy \
  --optimizer adam \
  --lr 0.001 \
  --max-samples 200 \
  --epochs 100 \
  --save models/'mnist model' \
  --plot