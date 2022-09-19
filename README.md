# DACOD360
Deadline-Aware Content Delivery Scheme for 360-degree Video Streaming over MEC-enabled C-RAN

### Prerequisites
- Install prerequisites (tested with Ubuntu 16.04, Tensorflow v1.1.0, TFLearn v0.3.1 and Selenium v2.39.0)
```
python setup.py
```

### Training
- To train a new model, put training data in `sim/cooked_traces`, then in `sim/` run
```
python rl_training.py
```

### Testing
- To test the trained model in simulated environment, first copy over the model to `test/models` and then in `test/` run `python get_video_sizes.py` and then run 
```
python main.py
```



