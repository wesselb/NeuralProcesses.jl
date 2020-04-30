using Pkg

Pkg.activate(".")
Pkg.resolve()
Pkg.instantiate()

using ConvCNPs
using ConvCNPs.Experiment
using Flux
using Flux.Tracker
using Stheno
using Distributions

# Construct data generator.
scale = 0.5f0
process = GP(stretch(eq(), 1 / 0.25), GPC())
data_gen = DataGenerator(
    process;
    batch_size=16,
    x=Uniform(-2, 2),
    num_context=DiscreteUniform(3, 50),
    num_target=DiscreteUniform(3, 50)
)

# Instantiate ConvCNP model.
arch = build_conv(4scale, 8, 64; points_per_unit=50f0, dimensionality=1)
model = convcnp_1d(arch; margin=4scale) |> gpu

# Train model.
opt = ADAM(5e-4)
epochs = 100
num_batches = 2048
bson = "model_eq.bson"
train!(
    model,
    data_gen,
    opt,
    bson=bson,
    batches_per_epoch=num_batches,
    epochs=epochs
)
