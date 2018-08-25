mutable struct GlobalMean <: CFModel
  μ::Float64
end

"""
    GlobalMean(dataset::CFDatasetAbstract)

Collaborative filtering baseline model that use a global mean to predict all the
ratings.

# Examples
```julia-repl
julia> model = Persa.GlobalMean(ds)
Persa.GlobalMean
  μ → 3.53…
julia> Persa.predict(model, 1, 1)
3.53…
```
"""
GlobalMean(dataset::T) where T<:CFDatasetAbstract = GlobalMean(mean(dataset))

train!(model::GlobalMean, dataset::T) where T<:CFDatasetAbstract = nothing

predict(model::GlobalMean, user::Int, item::Int) = model.μ

canpredict(model::GlobalMean, user::Int, item::Int) = true
