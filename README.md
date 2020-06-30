### Authorship
Experiments ran by T. A. Rupprecht March - August 2020

Models developed by github user kuangliu

Independent Component Layers conceptualized by the authors of
	"Rethinking the Usage of Batch Normalization and Dropout in the Training of Deep Neural Networks"

Independent Component Layers implemented by T. A. Rupprecht May 2020

### Script calls
## Training 
# without Independent Compnent Layers
python ./train.py --arch=resnet --depth=50 --save=./logs/ --dataset=cifar10 --save_name=resnet-test --lr=0.001 --epochs=1000 --cuda_num=6

# with static Independent Compnent Layers
python ./train.py --arch=resnet --depth=50 --save=./logs/ --dataset=cifar10 --save_name=resnet-test --lr=0.001 --epochs=1000 --cuda_num=6 --icl

# with dynamic Independent Component Layers
python ./train.py --arch=resnet --depth=50 --save=./logs/ --dataset=cifar10 --save_name=resnet-test --lr=0.001 --epochs=1000 --cuda_num=6 --icl --dynamic
