python main.py --dataset-path "./datasets" --dataset QFSD --model resnet12 --epochs 0 --manifold-mixup 500 --rotations --cosine --gamma 0.9 --milestones 100 --skip-epochs 450 --batch-size 128 --preprocessing ME --save-model "./checkpoint/QFSD3.pt1" --n-shot 1

python main.py --dataset-path "./datasets" --dataset QFSD --model resnet12 --epochs 0 --manifold-mixup 500 --rotations --cosine --gamma 0.9 --milestones 100 --skip-epochs 450 --batch-size 128 --preprocessing ME --save-model "./checkpoint/QFSD1.pt5" --n-shot 5

python main.py --dataset-path "./datasets" --dataset QFSD --model resnet12 --epochs 0 --manifold-mixup 500 --rotations --cosine --gamma 0.9 --milestones 100 --skip-epochs 450 --batch-size 128 --preprocessing ME --save-model "./checkpoint/QFSD2.pt5" --n-shot 5

python main.py --dataset-path "./datasets" --dataset QFSD --model resnet12 --epochs 0 --manifold-mixup 500 --rotations --cosine --gamma 0.9 --milestones 100 --skip-epochs 450 --batch-size 128 --preprocessing ME --save-model "./checkpoint/QFSD3.pt5" --n-shot 5

python main.py --dataset-path "./datasets" --dataset QFSD --model resnet12 --epochs 0 --load-model "/home/iim/eee/code/Mycode/easy/checkpoint/QFSD1.pt11" --save-features "./features/QFSDfeaturesAS1.pt1" --n-shots 1 --sample-aug 30

python main.py --dataset-path "./datasets" --dataset QFSD --model resnet12 --epochs 0 --load-model "/home/iim/eee/code/Mycode/easy/checkpoint/QFSD2.pt11" --save-features "./features/QFSDfeaturesAS2.pt1" --n-shots 1 --sample-aug 30

python main.py --dataset-path "./datasets" --dataset QFSD --model resnet12 --epochs 0 --load-model "/home/iim/eee/code/Mycode/easy/checkpoint/QFSD3.pt11" --save-features "./features/QFSDfeaturesAS3.pt1" --n-shots 1 --sample-aug 30

python main.py --dataset-path "./datasets" --dataset QFSD --model resnet12 --epochs 0 --load-model "/home/iim/eee/code/Mycode/easy/checkpoint/QFSD1.pt55" --save-features "./features/QFSDfeaturesAS1.pt5" --n-shots 5 --sample-aug 30

python main.py --dataset-path "./datasets" --dataset QFSD --model resnet12 --epochs 0 --load-model "/home/iim/eee/code/Mycode/easy/checkpoint/QFSD2.pt55" --save-features "./features/QFSDfeaturesAS2.pt5" --n-shots 5 --sample-aug 30

python main.py --dataset-path "./datasets" --dataset QFSD --model resnet12 --epochs 0 --load-model "/home/iim/eee/code/Mycode/easy/checkpoint/QFSD3.pt55" --save-features "./features/QFSDfeaturesAS3.pt5" --n-shots 5 --sample-aug 30
