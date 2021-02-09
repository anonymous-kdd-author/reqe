This is a reference implementation of Reqe model for multiple graphs.

Environment requirements:

Python >= 3.6
Pytorch >= 1.4.0
networkx
scikit-learn
CUDA

To test the model on DBLP dataset, simply run:

    python main.py --data_path data/dblp.txt --prob_path dblp.txt --graphs 18 --label_path data/label_dblp.txt -s 128  --sampling_size 10 --sampling -e 50
