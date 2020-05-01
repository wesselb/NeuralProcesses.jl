export load_checkpoints, checkpoint!, recent_model, best_model

struct Checkpoint
    model
    epoch
    loss_value
    loss_error
end

const MaybeCheckpoint = Union{Missing, Checkpoint}

mutable struct Checkpoints
    recent::MaybeCheckpoint
    top::Vector{Checkpoint}
    top_num::Integer
end

Checkpoints() = Checkpoints(missing, Vector{Checkpoint}(), 5)

Base.isless(c1::Checkpoint, c2::Checkpoint) = isless(c1.loss_value, c2.loss_value)

function load_checkpoints(bson)
    if !isfile(bson)
        return Checkpoints()
    else
        content = BSON.load(bson)
        return haskey(content, :checkpoints) ? content[:checkpoints] : Checkpoints()
    end
end

function checkpoint!(bson, model, epoch, loss_value, loss_error)
    checkpoints = load_checkpoints(bson)

    # Construct current checkpoint.
    current_checkpoint = Checkpoint(cpu(model), epoch, loss_value, loss_error)

    # Always update most recent model.
    checkpoints.recent = current_checkpoint

    # Update top models.
    extended_top = sort(vcat(checkpoints.top, current_checkpoint))
    checkpoints.top = extended_top[1:min(length(extended_top), checkpoints.top_num)]

    # Write changes.
    BSON.bson(bson, checkpoints=checkpoints)
end

recent_model(checkpoints::Checkpoints) = checkpoints.recent.model
recent_model(bson::String) = recent_model(load_checkpoints(bson))

best_model(checkpoints::Checkpoints, position::Integer=1) = checkpoints.top[position].model
best_model(bson::String, position::Integer=1) = best_model(load_checkpoints(checkpoints), position)
