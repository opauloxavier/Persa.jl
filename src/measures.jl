#Mean absolute error (MAE)
function mae(labels::Vector, predicted::Vector)
    index = .!isnan.(predicted)

    if isempty(index)
        return Inf
    end

    return mean(abs.(predicted[index] - labels[index]));
end
#Root mean squared error (RMSE)
function rmse(labels::Vector, predicted::Vector)
  s = 0.0

  index = .!isnan.(predicted)

  if isempty(index)
      return Inf
  end

  A = predicted[index] - labels[index];

  for a in A
    s += a*a
  end

  return sqrt(s / length(A))
end

#Coverage
coverage(predicted::Vector) = length(findall(r->!r, isnan.(predicted))) ./ length(predicted);

abstract type CFMetrics end

struct AccuracyMeasures <: CFMetrics
  mae::Float64
  rmse::Float64
  coverage::Float64
end

struct DecisionMetrics{T} <: CFMetrics
  confusion::Matrix{Int}
  map::Dict{T, Int}
  threshold::Real
end

struct RankAccuracy <: CFMetrics
  ndcg::Float64
  ndcg_k::Array{Tuple{Int, Float64}}
end

AccuracyMeasures(labels::Array, predict::Array) = AccuracyMeasures(mae(labels, predict), rmse(labels, predict), coverage(predict))

mae(measures::AccuracyMeasures) = measures.mae
rmse(measures::AccuracyMeasures) = measures.rmse
coverage(measures::AccuracyMeasures) = measures.coverage

function DecisionMetrics(model::CFModel, data_test::Array, preferences::RatingPreferences, threshold::Real)
    predicts = Persa.predict(model, data_test)

    return DecisionMetrics(data_test[:, 3], predicts, preferences, threshold::Real)
end

DecisionMetrics(model::CFModel, data_test::Array, preferences::RatingPreferences) = DecisionMetrics(model, data_test, preferences, recommendation(preferences))

function DecisionMetrics(labels::Array, predicts::Array, preferences::RatingPreferences, threshold::Real)
    @assert length(labels) == length(predicts) "Array of labels don't have the same size then predicts ones"
    @assert length(labels) > 0 "Labels/Predicts don't have elements"
    @assert threshold >= minimum(preferences) && threshold <= maximum(preferences) "Incorrect value of threshold"

    roundpredicts = round(predicts, preferences)

    preferencesmap = Dict{eltype(preferences), Int}()

    for i in sort(unique(preferences))
        preferencesmap[i] = length(preferencesmap) + 1
    end

    confusion = zeros(Int, size(preferences), size(preferences))

    for i = 1:length(labels)
        if !isnan(roundpredicts[i])
            confusion[preferencesmap[labels[i]], preferencesmap[roundpredicts[i]]] += 1
        end
    end

    return DecisionMetrics(confusion, preferencesmap, threshold::Real)
end

function recall(measures::DecisionMetrics{T}, class::T) where T
    if !haskey(measures.map, class)
        error("Invalid class")
    end

    index = measures.map[class]
    value = measures.confusion[index, index] ./ (sum(measures.confusion[index, :]))

    if isnan(value) || isinf(value)
        value = 0
    end

    return value
end

function precision(measures::DecisionMetrics{T}, class::T) where T
    if !haskey(measures.map, class)
        error("Invalid class")
    end

    index = measures.map[class]
    value = measures.confusion[index, index] ./ (sum(measures.confusion[:, index]))

    if isnan(value) || isinf(value)
        value = 0
    end

    return value
end

function f1score(measures::DecisionMetrics{T}, class::T) where T
    pre = precision(measures, class)
    rec = recall(measures, class)

    value = (2 * pre * rec) / (pre + rec)

    if isnan(value) || isinf(value)
        value = 0
    end

    return value
end

function recall(measures::DecisionMetrics)
    num = 0
    dem = 0

    for preference in keys(measures.map)
        if preference >= measures.threshold
            index = measures.map[preference]
            num += measures.confusion[index, index]
            dem += sum(measures.confusion[index, :])
        end
    end

    value = num / dem

    if isnan(value) || isinf(value)
        value = 0
    end

    return value
end

function precision(measures::DecisionMetrics)
    num = 0
    dem = 0

    for preference in keys(measures.map)
        if preference >= measures.threshold
            index = measures.map[preference]
            num += measures.confusion[index, index]
            dem += sum(measures.confusion[:, index])
        end
    end

    value = num / dem

    if isnan(value) || isinf(value)
        value = 0
    end

    return value
end

function f1score(measures::DecisionMetrics)
    pre = precision(measures)
    rec = recall(measures)

    value = (2 * pre * rec) / (pre + rec)

    if isnan(value) || isinf(value)
        value = 0
    end

    return value
end

function macrof1score(measures::DecisionMetrics)
    macroprecision = 0
    macrorecall = 0

    for preference in keys(measures.map)
        macroprecision += precision(measures, preference)
        macrorecall += recall(measures, preference)
    end

    macroprecision = macroprecision / length(keys(measures.map))
    macrorecall = macrorecall / length(keys(measures.map))

    value = (2 * macroprecision * macrorecall) / (macroprecision + macrorecall)

    if isnan(value) || isinf(value)
        value = 0
    end

    return value
end

struct ResultPredict{T} <: CFMetrics
  accuracy::AccuracyMeasures
  decision::DecisionMetrics{T}
  rank::RankAccuracy
end

function ResultPredict(model::CFModel, data_test::Array, preferences::RatingPreferences, threshold::Real)
  predicted = predict(model, data_test)

  acc = AccuracyMeasures(data_test[:,3], predicted)
  dec = DecisionMetrics(data_test[:,3], predicted, preferences, threshold)
  ran = RankAccuracy(predicted, data_test)

  return ResultPredict(acc, dec, ran)
end

ResultPredict(model::CFModel, data_test::Array, preferences::RatingPreferences) = ResultPredict(model, data_test, preferences, recommendation(preferences))

mae(measures::ResultPredict) = mae(measures.accuracy)
rmse(measures::ResultPredict) = rmse(measures.accuracy)
coverage(measures::ResultPredict) = coverage(measures.accuracy)

recall(measures::ResultPredict) = recall(measures.decision)
precision(measures::ResultPredict) = precision(measures.decision)
f1score(measures::ResultPredict) = f1score(measures.decision)

recall(measures::ResultPredict{T}, class::T) where T = recall(measures.decision, class)
precision(measures::ResultPredict{T}, class::T) where T = precision(measures.decision, class)
f1score(measures::ResultPredict{T}, class::T) where T = f1score(measures.decision, class)

macrof1score(measures::ResultPredict) = macrof1score(measures.decision)

ndcg(measures::ResultPredict) = ndcg(measures.rank)

function AccuracyMeasures(model::CFModel, data_test::Array)
  predicted = predict(model, data_test)

  return AccuracyMeasures(data_test[:,3], predicted)
end

ndcg(model::Persa.CFModel, ds_test::Array) = ndcg(Persa.predict(model, ds_test), ds_test)

function ndcg(predicts::Array, ds_test::Array)
    max_k = 0
    for user in unique(ds_test[:,1])
        index = findall(r->r == user, ds_test[:,1])

        if length(index) > max_k
            max_k = length(index)
        end
    end

    return ndcg(predicts, ds_test, max_k)
end

ndcg(model::Persa.CFModel, ds_test::Array, k::Int) = ndcg(Persa.predict(model, ds_test), ds_test, k)

function ndcg(predicts::Array, ds_test::Array, k::Int)
    ndcg = 0
    total = 0

    for user in unique(ds_test[:,1])
        index = findall(r->r == user, ds_test[:,1])

        if length(index) > 0
            perfect_list = sortrows(hcat(index, ds_test[index,3]), lt=(x,y)->isless(x[2],y[2]), rev = true)
            predict_list = sortrows(hcat(index, predicts[index]), lt=(x,y)->isless(x[2],y[2]), rev = true)

            ndcg += dcg(ds_test[predict_list[:,1], 3], k) ./ dcg(perfect_list[:,2], k)
            total += 1
        end
    end

    return ndcg ./ total
end


function dcg(values::Vector, k::Int)
    elements = length(values) > k ? k : length(values)

    dcg = 0

    for i = 1:elements
        dcg += values[i] ./ log2(1 + i)
    end

    return dcg
end

dcg(values::Vector) = dcg(values, length(values))

function RankAccuracy(predicts::Array, data_test::Array; ks::Vector{Int} = [5, 10, 20, 50])
    values = Array{Tuple{Int, Float64}}(length(ks))

    for i = 1:length(ks)
        values[i] = (ks[i], ndcg(predicts, data_test, ks[i]))
    end

    return RankAccuracy(ndcg(predicts, data_test), values)
end

ndcg(rank::RankAccuracy) = rank.ndcg

aval(model::CFModel, data_test::Array) = AccuracyMeasures(model, data_test)
aval(model::CFModel, data_test::Array, preferences::RatingPreferences) = ResultPredict(model, data_test, preferences)
aval(model::CFModel, data_test::Array, preferences::RatingPreferences, threshold::Real) = ResultPredict(model, data_test, preferences, threshold)

function DataFrame(result::AccuracyMeasures)
  df = DataFrame()

  df[:mae] = mae(result)
  df[:rmse] = rmse(result)
  df[:coverage] = coverage(result)

  return df
end

function DataFrame(result::DecisionMetrics)
    df = DataFrame()

    df[:recall] = recall(result)
    df[:precision] = precision(result)
    df[:f1score] = f1score(result)

    ratings = collect(keys(result.map))
    sort!(ratings)

    for rating in ratings
        df[Symbol("precision_$(rating)")] = precision(result, rating)
        df[Symbol("recall_$(rating)")] = recall(result, rating)
        df[Symbol("f1_$(rating)")] = f1score(result, rating)
    end

    df[:macrof1] = macrof1score(result)

    return df
end

function DataFrame(result::RankAccuracy)
    df = DataFrame()

    df[:ndcg] = ndcg(result)

    for (k, value) in result.ndcg_k
        df[Symbol("ndcg_$(k)")] = value
    end

    return df
end

DataFrame(result::ResultPredict) = hcat(DataFrame(result.accuracy), DataFrame(result.decision), DataFrame(result.rank))

function DataFrame(result::CFMetrics...)
  df = DataFrame()
  for i=1:length(result)
    df = vcat(df, DataFrame(result[i]))
  end

  return df
end

function Base.print(result::AccuracyMeasures)
  println("MAE - $(mae(result))")
  println("RMSE - $(rmse(result))")
  println("Coverage - $(coverage(result))")
end

function Base.print(result::DecisionMetrics)
    println("Recommendation (r >= $(result.threshold)):")

    println("Recall - $(recall(result))")
    println("Precision - $(precision(result))")
    println("F1-Score - $(f1score(result))")

    println("")

    ratings = collect(keys(result.map))
    sort!(ratings)

    for rating in ratings
        println("Rating $rating:")
        println("Precision - $(precision(result, rating))")
        println("Recall - $(recall(result, rating))")
        println("F1 - $(f1score(result, rating))")
        println("")
    end

    println("Global:")
    println("Macro F1 - $(macrof1score(result))")
end

function Base.print(result::RankAccuracy)
    df = DataFrame()

    println("NDCG - $(ndcg(result))")

    for (k, value) in result.ndcg_k
        println("NDCG $(k) - $(value)")
    end
end

function Base.print(result::ResultPredict)
    println("- Accuracy Metrics")
    print(result.accuracy)
    println("")
    println("- Decision Metrics")
    print(result.decision)
    println("")
    println("- Rank Metrics")
    print(result.rank)
end
