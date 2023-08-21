To train our model on ImageNet, run following command:  

Setting 1: Train our model without prototype loss:  
python main_distributed.py --dataset imagenet --arch resnet50 --epochs 200 --data /path/to/imagenet   
  
Setting 2: Train model with another augment_weight config:  
python main_distributed.py --dataset imagenet --arch resnet50 --epoch 200 --use-class-temperature --data /data/ImageNet/ILSVRC/Data/CLS-LOC/ --batch-size 128 --augment-weight 0.25  

Setting 3: Enable prototype loss with constant temperature :  
python main_distributed.py --dataset imagenet --arch resnet50 --epoch 200 --use-center --data /data/ImageNet/ILSVRC/Data/CLS-LOC/ --batch-size 128 --augment-weight 0.5   
 
  
Setting 4: Enable prototype loss with adaptive temperature :  
python main_distributed.py --dataset imagenet --arch resnet50 --epochs 200 --data /path/to/imagenet  --use-center  --use-class-temperature  

Setting 5 : Fine-tuning top classifier layer for pretrained model on ImageNet:  
python train_linear_classifier.py --loss_type LDAM --train_rule Reweight --arch resnet50 --lr 0.1 --dataset imagenet --pretrained_model path/to/model/model_best.pth --data_path /data/ImageNet/ILSVRC/Data/CLS-LOC/  

Setting 6 : Using TSC setting to fine-tuning ImageNet:  
python TSC_fine_tuning.py  --dataset imagenet  --pretrained ../path_to/model_best.pth --epochs 40 --schedule 20 30 --seed 0 -b 2048 --data  /data/ImageNet/ILSVRC/Data/CLS-LOC/

For testing accuracy of pretrained model with knn-evaluation:  
Download the pretrained checkpoint here:  
https://drive.google.com/u/0/uc?id=1XritMl3dYa9iW-TomaKU1XLQJVqgopMz&export=download  
and modify path of pretrained model in metric_evaluate.py file, then run:  
python metric_evaluate.py  
