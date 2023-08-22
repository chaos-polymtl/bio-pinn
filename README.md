# bio-pinn
Physics-Informed Neural Network to predict the reaction rates of a **BIO**diesel process

## Physics-informed Neural Network

Physics-informed Neural Network (or PINN for short) is a deep learning algorithm that solves differential equations (ODEs or PDEs). In this work, we aim to use PINN for the purpose of identifying a chemical reaction. The mass of chemical species is conserved throughout the reaction. The accumulation rate of a specie depends on the rate at which it is consumed.

$\frac{d C_A}{dt} = - r_A$

with $C_A$ the concentration of reagant A, $t$ the idependant variable (the time), and $r_A$ the consumption rate. 

The rate $r_A$ depends on a kinetic constant $k$ and a reaction order $\alpha$ that are a priori unknown, such as:

$r_A = k C_A^{\alpha}$

The PINN is able to perform a regression on data in order to solve de solution $C_A(t)$ and to identify the unknown kinetic parameters.

### 🧪 Biodiesel data (`pinn > biodiesel_data`)

The chemical reaction at study is the transesterification reaction in a microwave reactor where the glycerides in canola oil produce glycerol and biodiesel. First, the concentration data of glycerides are obtained via gas chromatography (GC-FID). The temperature is also measured within the reactor with an infrared sensor. Second, we build the architecture of the PINN. All the Python scripts are available in the `biodiesel_data`.

If you run the `get_results.py` script, you will automatically get the final predictions of the PINN. This script calls `numerical.py` in order to compare the PINN's predictions to a numerical solver.

In the `pinn` folder, we have the database and the scripts needed to train the PINN. Within this folder, 2 subfolders are present:

1. `data`: contains all the concentration data (from GC-FID) and the temperature data (from the infrared sensor).
2. `results`: contains all the predictions using the PINN and the numerical solver.

Within the same folder, we have the following scripts:

1. `data.py`: this script gathers the experimental data and creates the input tensor and the output tensor.
2. `pinn.py`: this script builds the architecture of the PINN. It uses the [PyTorch](https://pytorch.org/) library to do so.

- The class `PINeuralNet` (that uses the base class `torch.nn.Module`) allows to build the architecture, set up the parameters and defined the forward pass.
- The class Curiosity (named after the rover [Curiosity](https://mars.nasa.gov/msl/home/) from NASA that went on planet Mars to discover some wonders) trains the PINN model. The loss function is define in this class.

> Generates `loss.txt` to store the final evaluation of the loss function and `model.pt` to keep the PINN model in memory.

3. `main.py`: calls the two scripts presented above to train the PINN. It adapts the learning rate and the regularization parameters to optimize the training and the predictions.

4. `launch.sh`: allows to send requests to computer clusters in order to use their GPU ressources for longer training.

### 🦾 Artificial data (`pinn > artificial_data`)

The folder alongside the `biodiesel_data` one is the `artificial_data` folder. Using noisy artificial data from a simplified reaction, we verify the implementation of the PINN implemented for the biodiesel process. The reaction is:

$A \leftrightarrow C + B$ and

$C \leftrightarrow D$

In this folder, we have 3 Python scripts: `pinn.py` builds the PINN, `main.py` trains the PINN and `ode.py` solves numericaly the ODEs of species molar balance.

## Curve fit

Alongside the PINN algorithm, we implemented another non linear regressor for the purpose of comparing the performance of the PINN (a deep learning tool) with a more "classical" approach. The library [SciPy](https://scipy.org/) offers a function called `curve_fit`. Using the Trust Region algorithm (or `trf`), we try to identify the same kinetic parameters. The curves that the regressor tries to fit are the species concentration over time solved by a numerical integrator, which is a Runge-Kutta method.