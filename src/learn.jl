using Base: depwarn

abstract type CFModel

end

function predict(model::T, data::Array{Int, 2}) where T <: CFModel
  [ canpredict(model, data[i,1], data[i,2]) ? predict(model, data[i,1], data[i,2]) : NaN for i=1:size(data)[1]]
end

function predict(model::T, data::Array{Float64, 2}) where T <: CFModel
  [ canpredict(model, convert(Int, data[i,1]), convert(Int, data[i,2])) ? predict(model, convert(Int, data[i,1]), convert(Int, data[i,2])) : NaN for i=1:size(data)[1]]
end

predict(model::T, dataset::CFDatasetAbstract) where T <: CFModel = predict(model, dataset.file)

predict(model::T, data::DataFrame) where T <: CFModel = predict(model, Array(data))

function canpredict(model::CFModel, user::Int, item::Int)
    depwarn("canPredict is deprecated, use canPredict instead.", :canPredict)
    canPredict(model, user, item)
end

@deprecate canPredict(model::CFModel, user::Int, item::Int) canpredict(model, user, item)
