export CUDA_VISIBLE_DEVICES=0

DATA_DIR=./datasets
LOG_DIR=./exps_binconv

for DATASET in 'm4_daily' 'm4_weekly' 'tourism_monthly'
do
    for MODEL in 'binconv' 'bincov_prob'
    do
        python run.py --config config/m4/${DATASET}/${MODEL}.yaml --seed_everything 0  \
            --data.data_manager.init_args.path ${DATA_DIR} \
            --trainer.default_root_dir ${LOG_DIR}
    done
done
