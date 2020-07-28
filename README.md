<img src="https://github.com/wesselb/NeuralProcesses.jl/raw/master/loop.gif" width="800px" />

# NeuralProcesses.jl

NeuralProcesses.jl is a framework for composing
[Neural Processes](https://arxiv.org/abs/1807.01622) built on top of
[Flux.jl](https://github.com/FluxML/Flux.jl).

**Important:**
NeuralProcesses.jl requires [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) at
a version newer than v1.2.0.
Currently, this means that `master` is required for
[CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) and
[GPUArrays.jl](https://github.com/JuliaGPU/GPUArrays.jl): `]dev CUDA` and
`]dev GPUArrays`.

Contents:
- [Introduction](#introduction)
- [Manual](#manual)
- [State of the Package](#state-of-the-package)
- [Implementation Details](#implementation-details)

## Introduction

The setting of NeuralProcesses.jl is _meta-learning_. Meta-learning is concerned
with learning a map from data sets directly to predictive distributions.
Neural processes are a powerful class of parametrisations of this map based on
an _encoding_ of the data.

### Predefined Experimental Setup: `train.jl`

Eager to get started?!
The file `train.jl` contains an pre-defined experimental setup that gets
you going immediately!
Example:

```
$ julia --project=. train.jl --model convcnp --data matern52 --loss loglik --epochs 20
```

Here's what it can do:

```
$ julia --project=. train.jl --help
usage: train.jl --data DATA --model MODEL [--num-samples NUM-SAMPLES]
                [--batch-size BATCH-SIZE] --loss LOSS
                [--starting-epoch STARTING-EPOCH] [--epochs EPOCHS]
                [--evaluate] [--evaluate-iw] [--evaluate-no-iw]
                [--evaluate-num-samples EVALUATE-NUM-SAMPLES]
                [--evaluate-only-within] [--models-dir MODELS-DIR]
                [--bson BSON] [-h]

optional arguments:
  --data DATA           Data set: eq-small, eq, matern52,
                        noisy-mixture, weakly-periodic, sawtooth, or
                        mixture. Append "-noisy" to a data set to make
                        it noisy.
  --model MODEL         Model: conv[c]np, a[c]np, or [c]np. Append
                        "-global-{mean,sum}" to introduce a global
                        latent variable. Append
                        "-amortised-{mean,sum}" to use amortised
                        observation noise. Append "-het" to use
                        heterogeneous observation noise.
  --num-samples NUM-SAMPLES
                        Number of samples to estimate the training
                        loss. Defaults to 20 for "loglik" and 5 for
                        "elbo". (type: Int64)
  --batch-size BATCH-SIZE
                        Batch size. (type: Int64, default: 16)
  --loss LOSS           Loss: loglik, loglik-iw, or elbo.
  --starting-epoch STARTING-EPOCH
                        Set to a number greater than one to continue
                        training. (type: Int64, default: 1)
  --epochs EPOCHS       Number of epochs to training for. (type:
                        Int64, default: 20)
  --evaluate            Evaluate model.
  --evaluate-iw         Force to use importance weighting for the
                        evaluation objective.
  --evaluate-no-iw      Force to NOT use importance weighting for the
                        evaluation objective.
  --evaluate-num-samples EVALUATE-NUM-SAMPLES
                        Number of samples to estimate the evaluation
                        loss. (type: Int64, default: 4096)
  --evaluate-only-within
                        Evaluate with only the task of interpolation
                        within training range.
  --models-dir MODELS-DIR
                        Directory to store models in. (default:
                        "models")
  --bson BSON           Directly specify the file to save the model to
                        and load it from.
  -h, --help            show this help message and exit
```

### Example: The Convolutional Conditional Neural Process

As an example, below is an implementation of the
[Convolutional Conditional Neural Process](https://openreview.net/forum?id=Skey4eBYPS):

```julia
# The encoder maps into a function space, which is what `FunctionalCoder`
# indicates.
encoder = FunctionalCoder(
    # We cannot exactly represent a function, so we represent a discretisation
    # of the function instead. We use a discretisation of 64 points per unit.
    # The discretisation will span from the minimal context or target input
    # to the maximal context or target input with a margin of 1 on either side.
    UniformDiscretisation1D(64f0, 1f0),
    Chain(
        # The encoder is given by a so-called set convolution, which directly
        # maps the data set to the discretised functional representation. The
        # data consists of one channel. We also specify a length scale of
        # twice the inter-point spacing of the discretisation. The function
        # space that we map into is a reproducing kernel Hilbert space (RKHS),
        # and the length scale corresponds to the length scale of the kernel of
        # the RKHS. We also append a density channel, which ensures that the
        # encoder is injective.
        set_conv(1, 2 / 64f0; density=true),
        # The encoding will be deterministic. We could also use a stochastic
        # encoding.
        Deterministic()
    )
)

decoder = Chain(
    # The decoder first transforms the functional representation with a CNN.
    build_conv(
        1f0,  # Receptive field size
        10,   # Number of layers
        64,   # Number of channels
        points_per_unit =64f0,  # Density of the discretisation
        dimensionality  =1,     # This is a 1D model.
        num_in_channels =2,     # Account for density channel.
        num_out_channels=2      # Produce a mean and standard deviation.
    ),
    # Use another set convolution to map back from the space of the encoding
    # to the space of the data.
    set_conv(2, 2 / 64f0),
    # Predict means and variances.
    HeterogeneousGaussian()
)

convcnp = Model(encoder, decoder)
```

Then, after training, we can make predictions as follows:

```julia
means, lowers, uppers, samples = predict(
    convcnp,
    randn(Float32, 10),  # Random context inputs
    randn(Float32, 10),  # Random context outputs
    randn(Float32, 10)   # Random target inputs
)
```


## Manual

### Principles

#### Models

In NeuralProcesses.jl, models consists of an _encoder_ and a _decoder_.

```julia
model = Model(encoder, decoder)
```

An encoder takes in the data and produces an abstract representation of the
data.
A decoder then takes in this representation and produces a prediction at target
inputs.

#### Functional Representations and Coding

In the package, the three objects — the data, encoding, and prediction —
have a common representation. In particular, everything is represented as a
_function_: a tuple `(x, y)` that corresponds to the function `f(x[i]) = y[i]`
for all indices `i`.
Encoding and decoding, which we will collectively call _coding_, then become
_transformations of functions_.

Coding is implemented by the function `code`:

```julia
xz, z = code(encoder, xc, yc, xt)
```

Here `encoder` transforms the function `(xc, yc)`, the _context set_, into
another function `(xz, z)`, the abstract representation. The _target inputs_
`xt` express the desire that `encoder` _should_ (not must) output a function
that maps _from_ `xt`.
If indeed `xz == xt`, then the coding operation is called _complete_.
If, on the other hand, `xz != xt`, then the coding operation is called
_partial_.
Encoders and decoders are complete coders.
However, encoders and decoders are often composed from simpler coders, and these
coders could be partial.

#### Compositional Coder Design

Coders can be composed using `Chain`:

```julia
encoder = Chain(
    coder1,
    coder2,
    coder3
)
```

Coders can also be put in _parallel_ using `Parallel`.
For example, this is useful if an encoder should output multiple encodings,
e.g. in a multi-headed architecture:

```julia
encoder = Chain(
    ...,
    # Split the output into two parts, which can then be processed by two heads.
    Splitter(...),
    Parallel(
        ...,  # Head 1
        ...   # Head 2
    )
)
```

#### Coder Likelihoods

A coder should output either a _deterministic_ coding or a _stochastic_
coding, which can be achieved by appending a _likelihood_:

```julia
deterministic_coder = Chain(
    ...,
    Deterministic()
)

stochastic_coder = Chain(
    ...,
    HeterogeneousGaussian()
)
```

When a model is run, the output of the _encoder_ is _sampled_.
The resulting sample is then fed to the _decoder_.
In scenarios where the encoder outputs multiple encodings in parallel, it may be
desirable to concatenate those encodings into one big tensor which can then
be processed by the decoder.
This is achieved by prepending `Materialise()` to the decoder:

```julia
decoder = Chain(
    Materialise(),
    ...,
    HeterogenousGaussian()
)
```

The decoder outputs the prediction for the data.
In the above example, `decoder` produces means and variances at test inputs.

### Available Models for 1D Regression

The package exports constructors for a number of architectures from the
literature.

| Name | Constructor | Reference |
| :- | :- | :- |
| Conditional Neural Process | `cnp_1d` | [Garnelo, Rosenbaum, et al. (2018)](https://arxiv.org/abs/1807.01613) |
| Neural Process | `np_1d` | [Garnelo, Schwarz, et al. (2018)](https://arxiv.org/abs/1807.01622) |
| Attentive Conditional Neural Process | `acnp_1d` | [Kim et al. (2019)](https://arxiv.org/abs/1807.01622) |
| Attentive Neural Process | `anp_1d` | [Kim et al. (2019)](https://arxiv.org/abs/1807.01622) |
| Convolutional Conditional Neural Process | `convcnp_1d` | [Gordon et al. (2020)](https://openreview.net/forum?id=Skey4eBYPS) |
| Convolutional Neural Process | `convnp_1d` | [Foong et al. (2020)](https://arxiv.org/abs/2007.01332) |

### Building Blocks

The package provides various building blocks that can be used to compose
encoders and decoders.
For some building blocks, there is constructor function available that can be
used to more easily construct the block.
More information about a block can be obtained by using the built-in help
function, e.g. `?LayerNorm`.

#### Glue

| Type | Constructor | Description |
| :- | :- | :- |
| `Chain` | | Put things in sequence. |
| `Parallel` | | Put things in parallel. |

####  Basic Blocks

| Type | Constructor | Description |
| :- | :- | :- |
| `BatchedMLP` | `batched_mlp` | Batched MLP. |
| `BatchedConv` | `build_conv` | Batched CNN. |
| `Splitter` | | Split off a given number of channels. |
| `LayerNorm` | | Layer normalisation. |
| `MeanPooling` | | Mean pooling. |
| `SumPooling` | | Sum pooling. |

#### Advanced Blocks

| Type | Constructor | Description |
| :- | :- | :- |
| `Attention` | `attention` | Attentive mechanism. |
| `SetConv` | `set_conv` | Set convolution. |

#### Likelihoods

| Type | Constructor | Description |
| :- | :- | :- |
| `Deterministic` | | Deterministic output. |
| `FixedGaussian` | | Gaussian likelihood with a fixed variance. |
| `AmortisedGaussian` | | Gaussian likelihood with a fixed variance that is calculated from split-off channels. |
| `HeterogeneousGaussian` | | Gaussian likelihood with input-dependent variance. |

#### Coders

| Type | Constructor | Description |
| :- | :- | :- |
| `Materialise` | | Materialise a sample. |
| `FunctionalCoder` | | Code into a function space: make the target inputs a discretisation. |
| `UniformDiscretisation1D` | | Discretise uniformly at a given density of points. |
| `InputsCoder` | | Code with the target inputs. |
| `MLPCoder` | | Rho-sum-phi coder. |

### Data Generators

Models can be trained with data generators.
Data generators are callables that take in an integer (number of batches) and give
back an iterator that generates four tuples: context inputs, context outputs,
target inputs, and target outputs.
All tensors should be of rank three where the first dimension is the data
dimension, the second dimension is the feature dimension, and the third
dimension is the batch dimension.

Data generators can be constructed with `DataGenerator` which takes in an
underlying _stochastic process_.
[Stheno.jl](https://github.com/willtebbutt/Stheno.jl) can be used to build
Gaussian processes.
In addition, the package exports the following processes:

| Type | Description |
| :- | :- |
| `Sawtooth` | Sawtooth process.  |
| `BayesianConvNP` | A Convolutional Neural Process with a prior on the weights. |
| `Mixture` | A mixture of processes. |

### Training and Evaluation

Experimentation functionality is exported by `NeuralProcesses.Experiment`.

#### Running Models

A model can be run forward by calling it with three arguments:
context inputs, context outputs, and target inputs.
All arguments to models should be tensors of rank three where the first
dimension is the data dimension, the second dimension is the feature dimension,
and the third dimension is the batch dimension.

```julia
convcnp(
    # Use a batch size of 16.
    randn(Float32, 10, 1, 16),  # Random context inputs
    randn(Float32, 10, 1, 16),  # Random context outputs
    randn(Float32, 15, 1, 16)   # Random target inputs
)
```

For convenience, the package also exports the function `predict`, which
runs a model from inputs of type `Vector` and produces predictive means,
lower and upper credible bounds, and predictive samples.

```julia
means, lowers, uppers, samples = predict(
    convcnp,
    randn(Float32, 10),  # Random context inputs
    randn(Float32, 10),  # Random context outputs
    randn(Float32, 10)   # Random target inputs
)
```

#### Training

To train a model, use `train!`, which, amongst other things, requires a
loss function and optimiser.
Loss functions are described below, and
optimisers can be found in `Flux.Optimiser`;
for most applications, `ADAM(5e-4)` probably suffices.
After training, a model can be evaluated with `eval_model`.

See `train.jl` for an example of `train!`.

#### Losses

The following loss functions are exported:

| Function | Description |
| :- | :- |
| `loglik` | Biased estimate of the log-expected-likelihood. Exact for models with a deterministic encoder. |
| `elbo` | Neural process ELBO-style loss. |

Examples:

```julia
# 1-sample log-EL loss. This is exact for models with a deterministic encoder.
loss(xs...) = loglik(xs..., num_samples=1)   

# 20-sample log-EL loss. This is probably what you want if you are training
# a model with a stochastic encoder.
loss(xs...) = loglik(xs..., num_samples=20)

# 20-sample ELBO loss. This is an alternative to `loglik`.
loss(xs...) = elbo(xs..., num_samples=20)    
```

See `train.jl` for more examples.

#### Saving and Loading

After every epoch, the current model and top five best models are saved.
To file to which the model is written is determined by the keyword `bson`
of `train!`.
After training, the best model can be loaded with `best_model(path)`.


## State of the Package

The package is currently mostly a port from academic code.
There are a still a number of important things to do:

- **Support for 2D data:**
    The package is currently built around 1D tasks.
    We plan to add support for 2D data, e.g. images.
    This should not require big changes, but it should be implemented carefully.

- **Tests:**
    The important components of the package are tested, but test coverage is
    nowhere near where it should be.

- **Regression tests:**
    For the package, GPU performance is crucial, so regression tests are
    necessary.

- **Documentation:**
    Documentation needs to be improved.

## Implementation Details

### Automatic Differentiation

[Tracker.jl](https://github.com/FluxML/Tracker.jl) is used to automatically
compute gradients.
A number of custom gradients are implemented in `src/util.jl`.

### Parameter Handling

The package uses [Functors.jl](https://github.com/FluxML/Functors.jl) to handle
parameters, like [Flux.jl](https://github.com/FluxML/Flux.jl), and adheres
to [Flux.jl](https://github.com/FluxML/Flux.jl)'s principles.
This means that _only_ `AbstractArray{<:Number}`s are parameters.
Nothing else will be trained, not even `Float32` or `Float64` scalars.

To not train an array, wrap it with `NeuralProcesses.Fixed(array)`,
and unwrap it with `NeuralProcesses.unwrap(fixed)` at runtime.

### GPU Acceleration

CUDA support for depthwise separable convolutions (`DepthwiseConv` from
[Flux.jl](https://github.com/FluxML/Flux.jl)) is implemented in `src/gpu.jl`.

Loop fusion can cause issues on the GPU, so oftentimes computations are
unrolled.
