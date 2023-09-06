train:
	python train.py --data-file data/sensors-230-pollution.csv --epochs 100 --batch-size 16 --patience 10 --experiment-name my_experiment
eval:
	python eval.py --data-file data/sensors-230-pollution.csv --model-file results/my_experiment/trained_model.h5 --output-predictions
demo:
	python gradio_demo.py