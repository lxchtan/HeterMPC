cd analyse/coco-caption
SF='../../results/Paper/_Score_20220714.txt'
RF='../../results/Paper/Ground_Truth.txt'

python cal_score.py --score_file=${SF} --ref_file=${RF} --generate_file='../../results/Paper/HeterBERT.txt'
python cal_score.py --score_file=${SF} --ref_file=${RF} --generate_file='../../results/Paper/HeterBART.txt'
