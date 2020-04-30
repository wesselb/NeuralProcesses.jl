using Pkg

Pkg.activate(".")
Pkg.resolve()
Pkg.instantiate()

using BSON
using ConvCNPs
using ConvCNPs.Experiment
using Flux
using Stheno
using Distributions

# Construct data generator.
process = GP(stretch(eq(), 1 / 0.25), GPC())
data_gen = DataGenerator(
    process;
    batch_size=16,
    x=Uniform(-2, 2),
    num_context=DiscreteUniform(3, 50),
    num_target=DiscreteUniform(3, 50)
)

# Load ConvCNP model.
bson = "model_eq.bson"
@BSON.load bson model
model = model |> gpu

# Evaluate model.
eval_model(model, data_gen, 100; num_batches=10000)
