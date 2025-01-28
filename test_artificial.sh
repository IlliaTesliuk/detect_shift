#export METHOD=$1
export DST_DIR="exps2"

for ds_cov in 0.0 #0.5
do
    #for as in 0.0 0.02 0.04 0.06 0.08 0.10 0.15 0.20 0.25 0.30 0.40 0.50 0.60 0.70 0.80 0.90 1.00
    for as in 0.8 #0.0 0.5 1.0
    do
        echo $ds
        #export EXP_DIR="${DST_DIR}/results/artificial_${ds_cov}/kl_boot"
        #python test_method.py --method "kl_boot" --dataset "artificial" --ds_cov $ds_cov --exp_dir $EXP_DIR --attack_split $as --model "dt"
        #export EXP_DIR="${DST_DIR}/results/artificial_${ds_cov}/cdd"
        #python test_method.py --method "cdd" --dataset "artificial" --ds_cov $ds_cov --exp_dir $EXP_DIR --attack_split $as
        export EXP_DIR="${DST_DIR}/results/artificial_${ds_cov}/mmd"
        python test_method.py --method "mmd" --dataset "artificial" --exp_dir $EXP_DIR --attack_split $as --ds_cov $ds_cov
    done
done