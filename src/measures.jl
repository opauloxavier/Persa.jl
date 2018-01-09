#Mean absolute error (MAE)
function mae(labels::Vector, predicted::Vector)
    index = .!isnan.(predicted)

    return mean(abs.(predicted[index] - labels[index]));
end
#Root mean squared error (RMSE)
function rmse(labels::Vector, predicted::Vector)
  s = 0.0

  index = .!isnan.(predicted)
  A = predicted[index] - labels[index];

  for a in A
    s += a*a
  end

  return sqrt(s / length(A))
end

#Coverage
coverage(predicted::Vector) = length(find(r->!r, isnan.(predicted))) ./ length(predicted);

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

function recall{T}(measures::DecisionMetrics{T}, class::T)
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

function precision{T}(measures::DecisionMetrics{T}, class::T)
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

function f1score{T}(measures::DecisionMetrics{T}, class::T)
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
end

function ResultPredict(model::CFModel, data_test::Array, preferences::RatingPreferences, threshold::Real)
  predicted = predict(model, data_test)

  acc = AccuracyMeasures(data_test[:,3], predicted)
  dec = DecisionMetrics(data_test[:,3], predicted, preferences, threshold)

  return ResultPredict(acc, dec)
end

ResultPredict(model::CFModel, data_test::Array, preferences::RatingPreferences) = ResultPredict(model, data_test, preferences, recommendation(preferences))

mae(measures::ResultPredict) = mae(measures.accuracy)
rmse(measures::ResultPredict) = rmse(measures.accuracy)
coverage(measures::ResultPredict) = coverage(measures.accuracy)

recall(measures::ResultPredict) = recall(measures.decision)
precision(measures::ResultPredict) = precision(measures.decision)
f1score(measures::ResultPredict) = f1score(measures.decision)

recall{T}(measures::ResultPredict{T}, class::T) = recall(measures.decision, class)
precision{T}(measures::ResultPredict{T}, class::T) = precision(measures.decision, class)
f1score{T}(measures::ResultPredict{T}, class::T) = f1score(measures.decision, class)

macrof1score(measures::ResultPredict) = macrof1score(measures.decision)

recall(measures::ResultPredict) = recall(measures.decision)
precision(measures::ResultPredict) = precision(measures.decision)
f1score(measures::ResultPredict) = f1score(measures.decision)

function AccuracyMeasures(model::CFModel, data_test::Array)
  predicted = predict(model, data_test)

  return AccuracyMeasures(data_test[:,3], predicted)
end

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

DataFrame(result::ResultPredict) = hcat(DataFrame(result.accuracy), DataFrame(result.decision))

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

function Base.print(result::ResultPredict)
    println("- Accuracy Metrics")
    print(result.accuracy)
    println("")
    println("- Decision Metrics")
    print(result.decision)
end
