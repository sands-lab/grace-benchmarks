#!/usr/bin/env bash

mpirun \
-x NCCL_IB_DISABLE=1 -x NCCL_DEBUG=INFO -x PATH -x TF_CPP_MIN_LOG_LEVEL=3 \
-np 2 -H server1_ip:1,server2_ip:1 \
--display-map --display-allocation -map-by slot -bind-to none -nooversubscribe \
--mca btl_base_verbose 30 \
-mca pml ob1 -mca btl ^openib --tag-output --mca \
python \
grace-benchmarks/tensorflow/Classification/tf_cnn_benchmarks/tf_cnn_benchmarks.py \
--model=resnet20_v2 --data_name=cifar10 \
--batch_size=256 \
--weight_decay=0.0001 --optimizer=sgd \
--piecewise_learning_rate_schedule='0.1;163;0.01;245;0.001' --variable_update=horovod \
--summary_verbosity=1 --save_summaries_steps=10 --num_epochs=328 --eval_during_training_every_n_epochs=1 --num_eval_epochs=8 --data_dir=data/cifar10


mpirun \
-x NCCL_IB_DISABLE=1 -x NCCL_DEBUG=INFO -x PATH -x TF_CPP_MIN_LOG_LEVEL=3 \
-np 2 -H server1_ip:1,server2_ip:1 \
--display-map --display-allocation -map-by slot -bind-to none -nooversubscribe \
--mca btl_base_verbose 30 \
-mca pml ob1 -mca btl ^openib --tag-output --mca \
python \
grace-benchmarks/tensorflow/LanguageModeling/examples/PennTreebank/PTB-LSTM.py \
--datadir=data/PTB


mpirun \
-x NCCL_IB_DISABLE=1 -x NCCL_DEBUG=INFO -x PATH -x TF_CPP_MIN_LOG_LEVEL=3 \
-x WANDB_MODE=dryrun \
-np 2 -H server1_ip:1,server2_ip:1 \
--display-map --display-allocation -map-by slot -bind-to none -nooversubscribe \
--mca btl_base_verbose 30 \
-mca pml ob1 -mca btl ^openib --tag-output --mca \
python \
grace-benchmarks/tensorflow/Recommendation/NCF/ncf.py \
--batch-size=2097152 \
--checkpoint-dir /tmp \
--data data/ml-20m \
--epochs 10 --eval-after 0 --seed 20200107


mpirun \
-x NCCL_IB_DISABLE=1 -x NCCL_DEBUG=INFO -x PATH -x TF_CPP_MIN_LOG_LEVEL=3 \
-np 2 -H server1_ip:1,server2_ip:1 \
--display-map --display-allocation -map-by slot -bind-to none -nooversubscribe \
--mca btl_base_verbose 30 \
-mca pml ob1 -mca btl ^openib --tag-output --mca \
python \
grace-benchmarks/tensorflow/Segmentation/UNet_Industrial/main.py \
--unet_variant='tinyUNet' --activation_fn='relu' --exec_mode='train_and_evaluate' --iter_unit='batch' \
--batch_size=2 --warmup_step=10 --results_dir=results/segmentation_unet_DAGM2007/tmp --data_dir=data/DAGM2007 --dataset_name='DAGM2007' \
--dataset_classID=1 --data_format='NCHW' --use_auto_loss_scaling --learning_rate=1e-4 \
--learning_rate_decay_factor=0.8 --learning_rate_decay_steps=500 --rmsprop_decay=0.9 --rmsprop_momentum=0.8 \
--loss_fn_name='adaptive_loss' --weight_decay=1e-5 --weight_init_method='he_uniform' --augment_data \
--display_every=50 --debug_verbosity=0  --num_iter=250