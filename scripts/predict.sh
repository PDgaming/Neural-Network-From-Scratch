python cli.py predict \
  --load "models/mnist model.npz" \
  --dataset mnist_test \
  --architecture architecture.json \
  --input-size 784 \
  --output-size 10 \
  --row 154 \
  --task classification \
  --visualize
