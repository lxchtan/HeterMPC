export PYTHONPATH=`pwd`
export DGLBACKEND="pytorch"
LOGDIR=logs/`date +%Y%m%d`
if [ ! -d ${LOGDIR} ]; then
  mkdir -p ${LOGDIR}
fi

cal_group(){
  bash run_shell/H1_HeterBERT.sh ${SEED} > ${LOGDIR}/`date +%Y%m%d%H`_train_H1_S${SEED}.log  2>&1 
  bash run_shell/H2_HeterBART.sh ${SEED} > ${LOGDIR}/`date +%Y%m%d%H`_train_H2_S${SEED}.log  2>&1 
}

SEED=13 cal_group &
SEED=43 cal_group &
SEED=77 cal_group &
SEED=91 cal_group &