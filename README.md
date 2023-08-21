# bio-pinn
Physics-Informed Neural Network to predict the reaction rates of a BIOdiesel process

## Physics-informed Neural Network

### Biodiesel data

If you run the `get_results.py` script, you will automatically get the final predictions of the PINN. This script calls `numerical.py` in order to comparer the PINN's predictions to a numerical solver.

In the `pinn` folder, we have the database and the scripts needed to train the PINN. Within this folder, 2 subfolders are present:

1. `data`: contains all the concentration data (from GC-FID) and the temperature data (from the infrared sensor).
2. `results`: contains all the predictions using the PINN and the numerical solver.

Within the same folder, we have the following scripts:

1. `data.py`: this script gathers the experimental data and creates the input tensor and the output tensor.
2. `pinn.py`: this script builds the architecture of the PINN. It uses the PyTorch library to do so.

- The class `PINeuralNet` (that uses the base class `torch.nn.Module`) allows to build the architecture, set up the 4 parameters and define the forward pass.
- The class Curiosity (named after the rover [Curiosity](https://mars.nasa.gov/msl/home/) from NASA that went on planet Mars to discover some wonders) trains the PINN model. The loss function is define in this class.

> Generates `loss.txt` to store the final evaluation of the loss function and `model.pt` to keep the PINN model in memory.

3. `main.py`: calls the two scripts presented above to train the PINN. It adapts the learning rate and the regularization parameters to optimize the training and the predictions.

4. `launch.sh`: allows to send request to computer clusters in order to use their GPU ressources for longer training.

### Artificial data

## Curve fit

