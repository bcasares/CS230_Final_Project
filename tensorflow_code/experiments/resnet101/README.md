Uncoment the following line in model/model.py
~~~
# logits = resnet.build_resnet_(is_training, inputs, params)  # use resnet

~~~
|and run the experiment with:
~~~
|python train.py --model_dir experiments/renet101'
~~~

|from the following directory: CS230/Final_Project/CS230_Final_Project/tensorflow_code

