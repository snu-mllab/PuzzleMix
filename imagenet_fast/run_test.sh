DATA160=/data_large/readonly/ImageNet-Fast/imagenet-sz/160
DATA352=/data_large/readonly/ImageNet-Fast/imagenet-sz/352

NAME=puzzlemix

CONFIG1=configs/${NAME}/configs_fast_phase1.yml
CONFIG2=configs/${NAME}/configs_fast_phase2.yml
CONFIG3=configs/${NAME}/configs_fast_phase3.yml

PREFIX1=fast_phase1_${NAME}
PREFIX2=fast_phase2_${NAME}
PREFIX3=fast_phase3_${NAME}

OUT1=fast_train_phase1_${NAME}.out
OUT2=fast_train_phase2_${NAME}.out
OUT3=fast_train_phase3_${NAME}.out

EVAL1=fast_eval_phase1_${NAME}.out
EVAL2=fast_eval_phase2_${NAME}.out
EVAL3=fast_eval_phase3_${NAME}.out

END1=./trained_models/fast_phase1_${NAME}/checkpoint_epoch15.pth.tar
END2=./trained_models/fast_phase2_${NAME}/checkpoint_epoch40.pth.tar
END3=./trained_models/fast_phase3_${NAME}/checkpoint_epoch100.pth.tar

# training for phase 1
#python -u main_fast.py $DATA160 -c $CONFIG1 --output_prefix $PREFIX1 | tee $OUT1

# evaluation for phase 1
# python -u main_fast.py $DATA160 -c $CONFIG1 --output_prefix $PREFIX1 --resume $END1  --evaluate | tee $EVAL1

# training for phase 2
#python -u main_fast.py $DATA352 -c $CONFIG2 --output_prefix $PREFIX2 --resume $END1 | tee $OUT2

# evaluation for phase 2
# python -u main_fast.py $DATA352 -c $CONFIG2 --output_prefix $PREFIX2 --resume $END2 --evaluate | tee $EVAL2

# training for phase 3
#python -u main_fast.py $DATA352 -c $CONFIG3 --output_prefix $PREFIX3 --resume $END2 | tee $OUT3

# evaluation for phase 3
python -u main_test.py $DATA352 -c $CONFIG3 --output_prefix $PREFIX3 --resume $END3 --evaluate | tee $EVAL3

