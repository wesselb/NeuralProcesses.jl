<img src="https://github.com/wesselb/NeuralProcesses.jl/raw/master/loop.gif" width="800px" />

# NeuralProcesses.jl

NeuralProcesses.jl is a framework for composing
[Neural Processes](https://arxiv.org/abs/1807.01622) built on top of
[Flux.jl](https://github.com/FluxML/Flux.jl).

[NeuralProcesses.jl was presented at JuliaCon 2020 [link to video (7:41)].](https://www.youtube.com/watch?v=nq6X-w5xgLo)

See the [Neural Process Family](https://github.com/YannDubs/Neural-Process-Family) for code to reproduce the image experiments from [Convolutional Conditional Neural Processes (Gordon et al., 2020)](https://openreview.net/forum?id=Skey4eBYPS).

Contents:
- [Introduction](#introduction)
    - [Predefined Experimental Setup: `train.jl`](#predefined-experimental-setup-trainjl)
- [Manual](#manual)
    - [Principles](#principles)
    - [Available Models for 1D Regression](#available-models-for-1d-regression)
    - [Building Blocks](#building-blocks)
    - [Data Generators](#data-generators)
    - [Training and Evaluation](#training-and-evaluation)
- [Examples](#examples)
    - [The Conditional Neural Process](#the-conditional-neural-process)
    - [The Neural Process](#the-neural-process)
    - [The Attentive Neural Process](#the-attentive-neural-process)
    - [The Convolutional Conditional Neural Process](#the-convolutional-conditional-neural-process)
- [State of the Package](#state-of-the-package)
- [Implementation Details](#implementation-details)

## Introduction

The setting of NeuralProcesses.jl is _meta-learning_. Meta-learning is concerned
with learning a map from data sets directly to predictive distributions.
Neural processes are a powerful class of parametrisations of this map based on
an _encoding_ of the data.

### Predefined Experimental Setup: `train.jl`

Eager to get started?!
The file `train.jl` contains an predefined experimental setup that gets
you going immediately!
Example:

```
$ julia --project=. train.jl --model convcnp --data matern52 --loss loglik --epochs 20
```

Here's what it can do:

```
usage: train.jl --data DATA --model MODEL [--num-samples NUM-SAMPLES]
                [--batch-size BATCH-SIZE] --loss LOSS
                [--starting-epoch STARTING-EPOCH] [--epochs EPOCHS]
                [--evaluate] [--evaluate-iw] [--evaluate-no-iw]
                [--evaluate-num-samples EVALUATE-NUM-SAMPLES]
                [--evaluate-only-within] [--models-dir MODELS-DIR]
                [--bson BSON] [-h]

optional arguments:
  --data DATA           Data set: eq-small, eq, matern52, eq-mixture,
                        noisy-mixture, weakly-periodic, sawtooth, or
                        mixture. Append "-noisy" to a data set to make
                        it noisy.
  --model MODEL         Model: conv[c]np, corconvcnp, a[c]np, or
                        [c]np. Append "-global-{mean,sum}" to
                        introduce a global latent variable. Append
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

By default, parallel representations are combined with concatenation along the
channel dimension (see `Materialise` in the section below), but this is
readily extended to additional designs (see `?Materialise`).

#### Coder Likelihoods

A coder should output either a _deterministic_ coding or a _stochastic_
coding, which can be achieved by appending a _likelihood_:

```julia
deterministic_coder = Chain(
    ...,
    DeterministicLikelihood()
)

stochastic_coder = Chain(
    ...,
    HeterogeneousGaussianLikelihood()
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
| Attentive Conditional Neural Process | `acnp_1d` | [Kim, Mnih, et al. (2019)](https://openreview.net/forum?id=SkE6PjC9KX) |
| Attentive Neural Process | `anp_1d` | [Kim, Mnih, et al. (2019)](https://openreview.net/forum?id=SkE6PjC9KX) |
| Convolutional Conditional Neural Process | `convcnp_1d` | [Gordon, Bruinsma, et al. (2020)](https://openreview.net/forum?id=Skey4eBYPS) |
| Convolutional Neural Process | `convnp_1d` | [Foong, Bruinsma, et al. (2020)](https://arxiv.org/abs/2007.01332) |
| Gaussian Neural Process | `corconvcnp_1d` | [Bruinsma, Requeima et al. (2021)](https://openreview.net/forum?id=rzsDn7Vzxf) |

Download links for pretrained models are below.
The instructions for how a pretrained model can be run are as follows:

1. Download some [pretrained models](https://www.dropbox.com/s/ua40uc9ttzq9t18/models.tar.gz?dl=1).

2. Extract the models:

```bash
$ tar -xzvf models.tar.gz
```

3. Open Julia and load the model:

```julia
using NeuralProcesses, NeuralProcesses.Experiment, Flux

convnp = best_model("models/convnp-het/loglik/matern52.bson")
```

4. Run the model:

```julia
means, lowers, uppers, samples = predict(
    convnp,
    randn(Float32, 10),  # Random context inputs
    randn(Float32, 10),  # Random context outputs
    randn(Float32, 10)   # Random target inputs
)
```

#### Pretrained Models for [Foong, Bruinsma, et al. (2020)](https://arxiv.org/abs/2007.01332)

[Download link](https://www.dropbox.com/s/ua40uc9ttzq9t18/models.tar.gz?dl=1)

Interpolation results for the pretrained models are as follows:

##### `loglik`

| Model | `eq` | `matern52` | `noisy-mixture` | `weakly-periodic` | `sawtooth` | `mixture` |
|-|:-:|:-:|:-:|:-:|:-:|:-:|
| `cnp` | -1.09 | -1.17 | -1.28 | -1.34 | -0.16 | -1.17 |
| `acnp` | -0.83 | -0.93 | -1.00 | -1.29 | -0.17 | -1.09 |
| `convcnp` | -0.69 | -0.88 | -0.93 | -1.19 | 1.09 | -0.94 |
| `np-het` | -0.76 | -0.90 | -0.94 | -1.23 | -0.16 | -0.88 |
| `anp-het` | -0.53 | -0.74 | -0.66 | -1.17 | -0.10 | -0.63 |
| `convnp-het` | -0.34 | -0.61 | -0.59 | -1.01 | 2.24 | -0.40 |

##### `elbo`

| Model | `eq` | `matern52` | `noisy-mixture` | `weakly-periodic` | `sawtooth` | `mixture` |
|-|:-:|:-:|:-:|:-:|:-:|:-:|
| `np-het` | -0.34 | -0.66 | -0.66 | -1.21 | -0.12 | -0.71 |
| `anp-het` | -0.71 | -0.88 | -0.80 | -1.27 | -0.00 | -0.86 |
| `convnp-het` | -0.61 | -0.59 | -2.19 | -1.07 | 2.40 | -1.27 |

##### `loglik-iw`

| Model | `eq` | `matern52` | `noisy-mixture` | `weakly-periodic` | `sawtooth` | `mixture` |
|-|:-:|:-:|:-:|:-:|:-:|:-:|
| `np-het` | -0.28 | -0.57 | -0.47 | -1.20 | 0.32 | -0.59 |
| `anp-het` | -0.45 | -0.67 | -0.57 | -1.19 | -0.16 | -0.61 |
| `convnp-het` | -0.09 | -0.29 | -0.34 | -0.98 | 2.39 | -0.31 |

#### Pretrained Models for [Bruinsma, Requeima et al. (2021)](https://openreview.net/forum?id=rzsDn7Vzxf)

[Download link](https://www.dropbox.com/s/knlydai66aroorh/models.tar.gz?dl=1)

Interpolation results for the pretrained models are as follows:

##### `loglik`

| Model | `eq-noisy` | `matern52-noisy` | `noisy-mixture` | `weakly-periodic-noisy` | `sawtooth-noisy` | `mixture-noisy` |
|-|:-:|:-:|:-:|:-:|:-:|:-:|
| `convcnp` | -0.80 | -0.95 | -0.95 | -1.20 | 0.55 | -0.93 |
| `corconvcnp` | 0.70 | 0.30 | 0.96 | -0.47 | 0.42 | 0.10 |
| `anp-het` | -0.61 | -0.75 | -0.73 | -1.19 | 0.34 | -0.69 |
| `convnp-het` | -0.46 | -0.67 | -0.53 | -1.02 | 1.20 | -0.50 |

### Building Blocks

The package provides various building blocks that can be used to compose
encoders and decoders.
For some building blocks, there is a constructor function available that can be
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
| `SetConvPD` | `set_conv` | Set convolution for kernel functions. |

#### Likelihoods

| Type | Constructor | Description |
| :- | :- | :- |
| `DeterministicLikelihood` | | DeterministicLikelihood output. |
| `FixedGaussianLikelihood` | | Gaussian likelihood with a fixed variance. |
| `AmortisedGaussianLikelihood` | | Gaussian likelihood with a fixed variance that is calculated from split-off channels. |
| `HeterogeneousGaussianLikelihood` | | Gaussian likelihood with input-dependent variance. |

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
back an iterator that generates four-tuples: context inputs, context outputs,
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
| `Mixture` | Mixture of processes. |

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

## Examples

### The Conditional Neural Process

Perhaps the simplest member of the NP family is the
[Conditional Neural Process](http://proceedings.mlr.press/v80/garnelo18a.html)
(CNP).
CNPs employ a deterministic MLP-based encoder, and an MLP-based decoder.
As a first example, we provide an implementation of a simple CNP in the
framework:

```julia
# The encoder maps into a finite-dimensional vector space, and produces a
# global (deterministic) representation, which is then concatenated to every
# test point. We use a `Parallel` object to achieve this.
encoder = Parallel(
    # The `InputsEncoder` simply outputs the target locations. We `Chain` this
    # with a `DeterministicLikelihood` to form a complete coder.
    Chain(
        InputsCoder(),
        DeterministicLikelihood()
    ),
    Chain(
        # The representation is given by a deep-set network, which is
        # implemented with the `MLPCoder` object. This object receives two MLPs
        # upon construction, a pre-pooling network and post-pooling network,
        # and produces a vector representation for each context set in the
        # batch.
        MLPCoder(
            batched_mlp(
                dim_in    =dim_x + dim_y,
                dim_hidden=dim_embedding,
                dim_out   =dim_embedding,
                num_layers=num_encoder_layers
            ),
            batched_mlp(
                dim_in    =dim_embedding,
                dim_hidden=dim_embedding,
                dim_out   =dim_embedding,
                num_layers=num_encoder_layers
            )
        ),
        # The resulting representation is also chained with a
        # `DeterministicLikelihood` as we are interested in a conditional model.
        DeterministicLikelihood()
    )
)

# The CNP decoder is also MLP based. It first `materialises` the encoder output
# (concatenates the target inputs and context set representation), and then
# passes these through an MLP that outputs a mean and standard deviation at
# every target location.
decoder = Chain(
        # First, concatenate target inputs and context set representation. By
        # default, `Materialise` uses concatenation to combine the different
        # representations in a `Parallel` object, but alternative designs (e.g.,
        # summation or multiplicative flows) could also be considered in
        # NeuralProcesses.jl.
        Materialise(),
        # Pass the resulting representations through an MLP-based decoder. The
        # input dimensionality is the dimensionality of the target inputs plus
        # the dimensionality of the representation. The output dimension is
        # twice the output dimensionality, since we require a mean and standard
        # deviation for every location.
        batched_mlp(
            dim_in    =dim_x + dim_embedding,
            dim_hidden=dim_embedding,
            dim_out   =2dim_y,
            num_layers=num_decoder_layers
        ),
        # The `HeterogeneousGaussianLikelihood` automatically splits its inputs
        # in two along the feature dimension, and treats the first half as the
        # mean and second half as the standard deviation of a Gaussian
        # distribution.
        HeterogeneousGaussianLikelihood()
    )

cnp = Model(encoder, decoder)
```
Then, after training, we can make predictions as follows:

```julia
means, lowers, uppers, samples = predict(
    cnp,
    randn(Float32, 10),  # Random context inputs
    randn(Float32, 10),  # Random context outputs
    randn(Float32, 10)   # Random target inputs
)
```

### The Neural Process

[Neural Processes](https://arxiv.org/abs/1807.01622) (NP) extend CNPs by adding
a latent variable to the model.
This enables NPs to capture joint, non-Gaussian marginal distributions for
target sets, which in turn allows producing coherent samples.
Extending CNPs to NPs in NeuralProcceses.jl is extremely easy: we simply
replace the `DeterministicLikelihood` component of the `MLPCoder` with a
`HeterogenousGaussian`, and adjust the output dimension of the encoder to
produce both means and variances!

```julia
# The only change to the encoder is replacing the `DeterministicLikelihood`
# following the `MLPCoder` with a `HeterogenousGaussian`!
encoder = Parallel(
    Chain(
        InputsCoder(),
        DeterministicLikelihood()
    ),
    Chain(
        MLPCoder(
            batched_mlp(
                dim_in    =dim_x + dim_y,
                dim_hidden=dim_embedding,
                dim_out   =dim_embedding,
                num_layers=num_encoder_layers
            ),
            batched_mlp(
                dim_in    =dim_embedding,
                dim_hidden=dim_embedding,
                # Since `HeterogenousGaussian` splits its inputs along the
                # channel dimension, we increase the output dimension of the set
                # encoder accordingly.
                dim_out   =2dim_embedding,
                num_layers=num_encoder_layers
            )
        ),
        # This is the main change required to switch between a CNP and an NP.
        HeterogeneousGaussianLikelihood()
    )
)

# We can then reuse the previously defined decoder as is!
np = Model(encoder, decoder)
```

Note that typical NPs consider both a deterministic and latent representation.
This is easily achieved in NeuralProcesses.jl by adding an additional encoder to
the `Parallel` object (with a `DeterministicLikelihood`), and increasing the
decoder `dim_in` accordingly.
In this repo, the built-in NP model uses this form.
This example does not include a deterministic path to emphasise the ease of
switching between conditional and latent variable models in NeuralProcesses.jl.

### The Attentive Neural Processes

Next, we consider a more complicated model, and demonstrate how easy it is to
implement with NeuralProcesses.jl.
[Attentive Neural Processes](https://openreview.net/forum?id=SkE6PjC9KX) (ANPs)
extend NPs by considering an attentive mechanism for the deterministic
representation.
Attention comes built-in with NeuralProcesses.jl, and so we can deploy it within
a `Chain` or `Parallel` like other building blocks.
Below is an example implementation of an ANP with a deterministic attentive
representation, and a stochastic (Gaussian) global representation.

```julia
# The encoder now aggregates three separate representations:
#   (i) the target inputs, like the (C)NP,
#   (ii) a deterministic attentive representation, and
#   (iii) a stochastic global representation.
encoder = Parallel(
    # First, include the `InputsCoder` to represent the target set inputs.
    Chain(
        InputsCoder(),
        DeterministicLikelihood()
    ),
    # NeuralProcesses.jl uses a transformer-style multi-head architecture for
    # attention. It first embeds the inputs and outputs into a
    # finite-dimensional vector space with an MLP, and applies the attention in
    # the embedding space. The constructor requires the dimensionalities of the
    # inputs and outputs, the desired dimensionality of the embedding, and the
    # number of heads to employ (each head will use a
    # `div(dim_embedding, num_heads)`-dimensional embedding), and the number of
    # layers in the embedding MLPs. As ANPs employ attention for the
    # deterministic representations, this is chained with a
    # `DeterministicLikelihood`.
    Chain(
        attention(
            dim_x             =dim_x,
            dim_y             =dim_y,
            dim_embedding     =dim_embedding,
            num_heads         =num_encoder_heads,
            num_encoder_layers=num_encoder_layers
        ),
        DeterministicLikelihood()
    ),
    # The latent path uses the same form as for the NP.
    Chain(
        MLPCoder(
            batched_mlp(
                dim_in    =dim_x + dim_y,
                dim_hidden=dim_embedding,
                dim_out   =dim_embedding,
                num_layers=num_encoder_layers
            ),
            batched_mlp(
                dim_in    =dim_embedding,
                dim_hidden=dim_embedding,
                dim_out   =2dim_embedding,
                num_layers=num_encoder_layers
            )
        ),
        HeterogeneousGaussianLikelihood()
    )
)

# The decoder for the ANP is again MLP-based, and so has the same form as the
# (C)NP decoder. The only required change is to account for the dimensionality
# of the latent representation.
decoder = Chain(
    Materialise(),
    batched_mlp(
        dim_in    =dim_x + 2dim_embedding,
        dim_hidden=dim_embedding,
        dim_out   =num_noise_channels,
        num_layers=num_decoder_layers
    ),
    noise
)

anp = Model(encoder, decoder)
```

### The Convolutional Conditional Neural Process

As a final example, we consider the
[Convolutional Conditional Neural Process](https://openreview.net/forum?id=Skey4eBYPS)
(ConvCNP).
The key difference between the ConvCNP and other NPs in terms of implementation
is that it encodes the data into an infinite-dimensional function space, rather
than a finite-dimensional vector space.
This is handled in NeuralPeocesses.jl with `FunctionalCoder`s, which, in
addition to complete coders, also expect `Discretisation` objects on construction.
Below is an implementation of the
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
        DeterministicLikelihood()
    )
)

decoder = Chain(
    # The decoder first transforms the functional representation with a CNN.
    build_conv(
        4f0,  # Receptive field size
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
    HeterogeneousGaussianLikelihood()
)

convcnp = Model(encoder, decoder)
```

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
