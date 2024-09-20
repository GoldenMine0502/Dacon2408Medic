python main.py --criterion msle > log_msle.txt
python main.py > log.txt
python main.py --learning_rate 1e-5 > log_1e-5.txt
python main.py --criterion threshold_penalty > log_loss_penalty.txt
python main.py --criterion threshold_nolearn > log_loss_nolearn.txt
