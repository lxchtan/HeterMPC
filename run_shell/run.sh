export PYTHONPATH=`pwd`
export DGLBACKEND="pytorch"
LOGDIR=logs/`date +%Y%m%d`
if [ ! -d ${LOGDIR} ]; then
  mkdir -p ${LOGDIR}
fi

cal_group(){
  bash run_shell/H1_HeterBERT.sh ${SEED} > ${LOGDIR}/`date +%Y%m%d%H`_train_H1_S${SEED}.log  2>&1 
  bash run_shell/H2_HeterBART.sh ${SEED} > ${LOGDIR}/`date +%Y%m%d%H`_train_H2_S${SEED}.log  2>&1 
  bash run_shell/H3_FHeterBART.sh ${SEED} > ${LOGDIR}/`date +%Y%m%d%H`_train_H3_S${SEED}.log  2>&1 
  bash run_shell/M3-1_FHeterBART.sh ${SEED} > ${LOGDIR}/`date +%Y%m%d%H`_train_M3-1_S${SEED}.log  2>&1 
  bash run_shell/M3-2_EM_FHerterBART.sh ${SEED} > ${LOGDIR}/`date +%Y%m%d%H`_train_M3-2_S${SEED}.log  2>&1 
}

SEED=13 cal_group &
SEED=43 cal_group &
SEED=77 cal_group &
SEED=91 cal_group &
