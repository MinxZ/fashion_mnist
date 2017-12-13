models='Xception'
for model in $models
do
python train.py \
  --model $model \
  --lr 1e-02 \
  --optimizer "Adam"

python train.py \
  --model $model \
  --lr 5e-04 \
  --optimizer "SGD"
done
