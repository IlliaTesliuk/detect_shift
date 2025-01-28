#export METHOD=$1
export DST_DIR="exps2"
#export DATASET=$1
#for as in 0.0 0.02 0.04 0.06 0.08 0.10 0.15 0.20 0.25 0.30 0.40 0.50 0.60 0.70 0.80 0.90 1.00 # adult_income
#for as in 0.0 0.02 0.04 0.06 0.08 0.10 0.15 0.20 0.25 0.30 0.40 0.50 0.60 0.70 0.80 0.90 1.00 # bank_marketing

#for as in 0.00 0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.50 0.60 0.70 0.80 0.90 1.00 # credit_default

#for as in 0.0 0.02 0.04 0.06 0.08 0.10 0.15 0.20 0.25 0.30 0.40 0.50 0.60 0.70 0.80 0.90 1.00 # credit_risk
#do
#    echo $ds
#    export EXP_DIR="${DST_DIR}/results/${DATASET}/kl_boot"
#    python test_method.py --method "kl_boot" --dataset ${DATASET} --exp_dir $EXP_DIR --attack_split $as --model "dt"
#    export EXP_DIR="${DST_DIR}/results/${DATASET}/cdd"
#    python test_method.py --method "cdd" --dataset ${DATASET} --exp_dir $EXP_DIR --attack_split $as
#done

for ds in "adult_income" "cifar10_binary" "mnist_binary"
do
    for as in 0.80 #0.00 0.50 1.00
    do
        echo $ds
        export EXP_DIR="${DST_DIR}/results/${DATASET}/mmd"
        python test_method.py --method "mmd" --dataset $ds --exp_dir $EXP_DIR --attack_split $as
    done
done