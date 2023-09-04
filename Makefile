train:
	python train.py --data-file data/sensors-230-pollution.csv --target-column target_column --epochs 100 --batch-size 32 --test-size 0.2 --validation-split 0.2 --patience 10 --experiment-name my_experiment
eval:
	python eval.py --data-file data/sensors-230-pollution.csv --model-file trained_model.h5 --target-column target_column
demo:
	python gradio_demo.py