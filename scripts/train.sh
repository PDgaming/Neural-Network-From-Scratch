python cli.py train \
  --dataset wdbc \
  --architecture architecture.json \
  --loss binary_crossentropy \
  --optimizer adam \
  --lr 0.001 \
  --epochs 500 \
  --save models/'wdbc model' \
  --plot