python cli.py predict \
  --load models/wine.npz \
  --dataset wine \
  --architecture architecture.json \
  --input-size 13 \
  --output-size 3 \
  --input "12.85,3.27,2.58,22,106,1.65,0.6,0.6,0.96,5.58,0.87,2.11,570"