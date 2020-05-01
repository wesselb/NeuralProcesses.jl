export checkpoint, recent_model, best_model

struct Checkpoint
    model
    epoch
    loss_value
    loss_error
end

const MaybeCheckpoint = Union{Missing, Checkpoint}

struct Checkpoints
    recent::MaybeCheckpoint
    top::Vector{Checkpoint}
    top_num::Integer
end

Checkpoints() = Checkpoints(missing, Vector{Checkpoint}(), 5)

Base.isless(c1::Checkpoint, c2::Checkpoint) = isless(c1.loss_value, c2.loss_value)

function _load_checkpoints(bson)
    if !isfile(bson)
        return Checkpoints()
    else
        content = BSON.load(bson)
        return haskey(content, :checkpoints) ? content[:checkpoints] : Checkpoints()
    end
end

function checkpoint(bson, model, epoch, loss_value, loss_error)
    checkpoints = _load_checkpoints(bson)

    # Construct current checkpoint.
    current_checkpoint = Checkpoint(cpu(model), epoch, loss_value, loss_error)

    # Update most recent model.
    if ismissing(checkpoints.recent) || epoch == 1
        checkpoints.recent = current_checkpoint
    end

    # Update top models.
    checkpoints.top = sort(vcat(checkpoints, current_checkpoint))[1:content.top_num]

    # Write changes.
    BSON.bson(bson, checkpoints=checkpoints)
end

recent_model(bson) = _load_checkpoints(bson).recent.model |> gpu
best_model(bson, position::Integer=1) = _load_checkpoints(bson).top[position].model |> gpu
