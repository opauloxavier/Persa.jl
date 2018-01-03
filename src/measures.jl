#Mean absolute error (MAE)
mae(labels, predicted) = mean(abs.(predicted[find(r -> r > 0, predicted),1] - labels[find(r -> r > 0, predicted),1]));

#Root mean squared error (RMSE)
function rmse(labels, predicted)
  s = 0.0

  A = predicted[find(r -> r > 0, predicted),1] - labels[find(r -> r > 0, predicted),1];

  for a in A
    s += a*a
  end
  return sqrt(s / length(A))
end

#Coverage
coverage(predicted) = length(find(r-> r > 0, predicted[:,1])) ./ length(predicted[:,1]);

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

    return DecisionMetrics(data_test[:, 3], round(predicts, preferences), unique(preferences), threshold::Real)
end

DecisionMetrics(model::CFModel, data_test::Array, preferences::RatingPreferences) = DecisionMetrics(model, data_test, preferences, recommendation(preferences))

function DecisionMetrics(labels::Array, predict::Array, preferences::Array, threshold::Real)
    preferencesmap = Dict{eltype(preferences), Int}()

    for i in preferences
        preferencesmap[i] = length(preferencesmap) + 1
    end

    confusion = zeros(Int, length(preferences), length(preferences))

    for i = 1:length(labels)
        if !isnan(predict[i])
            confusion[preferencesmap[labels[i]], preferencesmap[predict[i]]] += 1
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


struct ResultPredict <: CFMetrics
  accuracy::AccuracyMeasures
  decision::DecisionMetrics
end

function ResultPredict(model::CFModel, data_test::Array, threshold::Real)
  predicted = predict(model, data_test)
  acc = AccuracyMeasures(data_test[:,3], predicted)
  dec = DecisionMetrics(data_test[:,3], predicted, threshold)

  return ResultPredict(acc, dec)
end

mae(measures::ResultPredict) = mae(measures.accuracy)
rmse(measures::ResultPredict) = rmse(measures.accuracy)
coverage(measures::ResultPredict) = coverage(measures.accuracy)

recall(measures::ResultPredict) = recall(measures.decision)
precision(measures::ResultPredict) = precision(measures.decision)
f1score(measures::ResultPredict) = f1score(measures.decision)

function AccuracyMeasures(model::CFModel, data_test::Array)
  predicted = predict(model, data_test)

  return AccuracyMeasures(data_test[:,3], predicted)
end

aval{T <: CFModel}(model::T, data_test::Array) = AccuracyMeasures(model, data_test)
aval{T <: CFModel}(model::T, data_test::Array, threshold::Number) = ResultPredict(model, data_test, threshold)

function Base.print(result::ResultPredict)
  print(result.accuracy)
  print(result.decision)
end

DataFrame(result::ResultPredict) = hcat(DataFrame(result.accuracy), DataFrame(result.decision))

function Base.print(result::AccuracyMeasures)
  println("MAE - $(mae(result))")
  println("RMSE - $(rmse(result))")
  println("Coverage - $(coverage(result))")
end

function DataFrame(result::AccuracyMeasures)
  df = DataFrame()
  df[:mae] = result.mae
  df[:rmse] = result.rmse
  df[:coverage] = result.coverage
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

function DataFrame(result::CFMetrics...)
  df = DataFrame()
  for i=1:length(result)
    df = hcat(df, DataFrame(result[i]))
  end

  return df
end
