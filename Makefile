process:
	python process.py data/saveecobot_15705.csv train.parquet test.parquet
train:
	python train.py --epochs 100 --batch-size 16 --patience 10 --experiment-name my_experiment
eval:
	python eval.py --data-file data/test.parquet --model-file results/my_experiment/trained_model.keras --output-predictions