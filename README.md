# Conditional GAN
Generating FanshionMNIST-like data with Conditional GAN.

<p align="center">
<img src="https://github.com/simoneVU/CGAN/blob/main/images/GD.gif" width="500" height="100" />
</p>

## Setup the environment
Open the terminal and create a conda environment by running `conda create --name py310 python=3.10`.

Then, activate the conda environment by running `conda activate py310`.

Once the environment is active, install the packages in the *requirements.txt* file by running `pip install -r requirements.txt`. 

## Running the CGAN train script
### Conditional GAN
The mathematical formula defining the CGAN is $\underset{G}{min}\underset{D}{max}V(D,G) = \mathop{\mathbb{E}}_{\boldsymbol{x}\sim p(\boldsymbol{x})}\[log D(\boldsymbol{x}|\boldsymbol{y})]+\mathop{\mathbb{E}}\_{\boldsymbol{x}\sim p(\boldsymbol{z})}\[ log(1-D(G(\boldsymbol{z}|\boldsymbol{y})))\]$

The following image shows the training loss curves of the Discriminator and Generator over 250 epochs of training. 

<p align="center">
<img src="https://github.com/simoneVU/CGAN/blob/main/images/loss_curve.png" width="600" height="500" />
</p>

Furthermore, from the loss curves above it is possible to see that the Generator improves its loss after around 50 epochs, while, the Discriminator slighly get worse. Hence, the ability of the Discriminator ogf disthinguishing newly generated images from the Generator slighlty decrease by reaching a loss of 0.7 on average. In turn, this task for the Discriminator becomes harder the more the Generator decreases its loss. This behavior is also noticeable at the start of training when the Discriminator does not know how to generate images. Thus, the discriminator loss is very low since it can recognize the real images vs the fake images easily.

### Traning the model
To train the model use the script `train.py`. The script already loads the FashionMNIST dataset from *~/.pytorch/F_MNIST_data* and transform it appropritely. The training of the model can be run with some optional terminal arguments by running `python train.py --<arg_name> <arg_value>`. The terminal argument that can be passed are the following:

- **n_epochs** (int): number of traning epochs for the generator and discriminator. Default value is 250.
- **batch_size** (int): batch size. Default value is 64.
- **lr** (float): learning rate for the Adam optimizer. Default value is 0.0004.
- **latent_dim** (int): the dimension of the latent space where the initial samples for the generator are drawn from. Default value 100.
- **img_dim** (int): each image dimension. E.g. \[img_dim, img_dim \] is an image of width and height equal to img_dim. Default value is 28 (specifically for FashionMNIST).
- **num_classes** (int): number of dataset classes. Default value is set to 10 (specifically for FashionMNIST).

Example of run a training for the FashionMNIST dataset: `python train.py --n_epochs 10 --lr 1e-3 --latent_dim 400 --batch_size 32`

### Testing the model

To test the model run `python test.py --latent_dim <default> --num_classes <default> --img_dim <default>`, where the command line argument are optional. However, not that the `num_classes` and the `img_dim` are dataset dependent. Hence, in this case they should not be change in order to run it on the FashionMNIST dataset (the default dataset). The output of the model testing is saved in *images/comparisons*. An example of one output for label 1 is the following

<p align="center">
<img src="https://github.com/simoneVU/CGAN/blob/main/images/comparisons/comparison_1.png" width="400" height="300" />
<img src="https://github.com/simoneVU/CGAN/blob/main/images/comparisons/comparison_2.png" width="400" height="300" />
</p>

## Running the Flask application 
In order to run the flask application it is first needed to link it with the flask command. Hence, open in the terminal run `export FLASK_APP=flask_app.py` (for MAC users) or  `set FLASK_APP=flask_app.py` (for windows users). Then, start the Flask server by running  `flask run` in the folder containing the `flask_app.py` script. Then, in the terminal should appear the following

```
 * Serving Flask app 'flask_app.py'
 * Debug mode: off
WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
 * Running on http://127.0.0.1:5000
Press CTRL+C to quit
```

Now, the Flask server is up and running at address *http://127.0.0.1:5000*. In order to run the flask_app.py, run the following command  `curl --silent --show-error --fail 'http://localhost:5000/predict' -d 'label=<label>' --output <filepath>.jpg`, where `<label>` is a placeholder for any label of choice from the FashionMNIST dataset and `<filepath>` is a placeholder for the name where the image output is saved. Therefore, to generate an image of a ankle boot  type  `curl --silent --show-error --fail 'http://localhost:5000/predict' -d 'label=Ankle boot' --output images/flask_generated/ankleboot.jpg`. The available labels for FashionMNIST are the following: *T-shirt/top*, *Trouser*, *Pullover*,*Dress*,*Coat*,*Sandal*,*Shirt*,*Sneaker*,*Bag* and *Ankle boot*.

## Unit tests
The test for the generator network are contained in the file *unit_tests/unit_test_generator.py*. In order to run the entire test suit run, from the project root folder, run `pytest unit_tests/unit_test_generator.py` in the terminal. Furthermore, to run only a specific test run `pytest unit_tests/unit_test_generator.py::TestGenerator::<test_func>`. The available tests are:

- **test_shape**: tests whether the input noise tensor shape for the generator matches the output image shape
- **test_output_range**: tests that the pixels in the output image are all between -1 and 1 and not *NaN*.
- **test_device_moving**: tests the moving of the model from CPU to GPU if GPU is available and whether the outputs are the same.
- **test_batch_independence**: tests whether the batch samples influence each others when fed to the model.
- **test_all_parameters_updated**: check if any dead sub-graphs exist by checking the model parameters gradients after the optimization step

