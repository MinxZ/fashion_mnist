models='Xception'
for model in $models
do
python train.py \
  --model $model \
  --lr 1e-02 \
  --optimizer "Adam"

python train.py \
  --model $model \
  --lr 1e-03 \
  --optimizer "Adam"

python train.py \
  --model $model \
  --lr 1e-04 \
  --optimizer "Adam"

python train.py \
  --model $model \
  --lr 1e-04 \
  --optimizer "SGD"

python train.py \
  --model $model \
  --lr 1e-05 \
  --optimizer "SGD"
done

