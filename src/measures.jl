using MLBase

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

abstract type CFMetrics

end

struct AccuracyMeasures <: CFMetrics
  mae::Float64
  rmse::Float64
  coverage::Float64
end

struct DecisionMetrics <: CFMetrics
  roc::MLBase.ROCNums
end

AccuracyMeasures(labels::Array, predict::Array) = AccuracyMeasures(mae(labels, predict), rmse(labels, predict), coverage(predict))

mae(measures::AccuracyMeasures) = measures.mae
rmse(measures::AccuracyMeasures) = measures.rmse
coverage(measures::AccuracyMeasures) = measures.coverage

DecisionMetrics(labels::Array, predict::Array, threshold::Number) = DecisionMetrics(roc(labels .>= threshold, predict .>= threshold))

recall(measures::DecisionMetrics) = MLBase.recall(measures.roc)
precision(measures::DecisionMetrics) = MLBase.precision(measures.roc)
f1score(measures::DecisionMetrics) = MLBase.f1score(measures.roc)

struct ResultPredict <: CFMetrics
  accuracy::AccuracyMeasures
  decision::DecisionMetrics
end

function ResultPredict(model::CFModel, data_test::Array, threshold::Number)
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

function Base.print(result::DecisionMetrics)
  println("Recall - $(recall(result))")
  println("Precision - $(precision(result))")
  println("F1-Score - $(f1score(result))")
end

function DataFrame(result::DecisionMetrics)
  df = DataFrame()
  df[:recall] = recall(result)
  df[:precision] = precision(result)
  df[:f1score] = f1score(result)
  return df
end

function DataFrame(result::CFMetrics...)
  df = DataFrame()
  for i=1:length(result)
    df = hcat(df, DataFrame(result[i]))
  end

  return df
end

struct ClassDecisionMeasures <: CFMetrics
    precision::Dict
    recall::Dict
    f1::Dict
    macrof1::AbstractFloat
end

aval{T <: CFModel}(model::T, data_test::Array, preferences::RatingPreferences) = ClassDecisionMeasures(model, data_test, preferences)

function ClassDecisionMeasures(model::CFModel, data_test::Array, preferences::RatingPreferences)
  predicts = Persa.predict(model, data_test)

  #TODO: Better this code
  values = zeros(eltype(preferences), length(predicts), 1)

  i = 0
  for i = 1:length(predicts)
      if !isnan(predicts[i])
          values[i] = round(predicts[i], preferences)
      end
  end

  nonnanindex = find(r->!isnan(r), predicts)

  return ClassDecisionMeasures(data_test[nonnanindex, 3], values[nonnanindex], unique(preferences))
end

function ClassDecisionMeasures(labels::Array, predict::Array, preferences::Array)
    preferencesmap = Dict{eltype(preferences), Int}()

    for i in preferences
        preferencesmap[i] = length(preferencesmap) + 1
    end

    confusion = zeros(Int, length(preferences), length(preferences))

    for i = 1:length(labels)
        confusion[preferencesmap[labels[i]], preferencesmap[predict[i]]] += 1
    end

    precisionmap = Dict{eltype(preferences), AbstractFloat}()
    recallmap = Dict{eltype(preferences), AbstractFloat}()
    f1map = Dict{eltype(preferences), AbstractFloat}()

    for preference in preferences
        index = preferencesmap[preference]

        precision = confusion[index, index] ./ (sum(confusion[:, index]))
        recall = confusion[index, index] ./ (sum(confusion[index, :]))

        if isnan(precision) || isinf(precision)
            precision = 0
        end

        if isnan(recall) || isinf(recall)
            recall = 0
        end

        f1 = (2 * precision * recall) / (precision + recall)

        if isnan(f1) || isinf(f1)
            f1 = 0
        end

        precisionmap[preference] = precision
        recallmap[preference] = recall
        f1map[preference] = f1
    end

    macroprecision = sum(values(precisionmap)) / length(preferences)
    macrorecall = sum(values(recallmap)) / length(preferences)

    macrof1 = (2 * macroprecision * macrorecall) / (macroprecision + macrorecall)

    return ClassDecisionMeasures(precisionmap, recallmap, f1map, macrof1)
end

function DataFrame(result::ClassDecisionMeasures)
    df = DataFrame()
    
    ratings = collect(keys(result.precision))
    sort!(ratings)

    for rating in ratings
        df[Symbol("precision_$(rating)")] = result.precision[rating]
        df[Symbol("recall_$(rating)")] = result.recall[rating]
        df[Symbol("f1_$(rating)")] = result.f1[rating]
    end

    df[:macrof1] = result.macrof1
    return df
end

function Base.print(result::ClassDecisionMeasures)
    ratings = collect(keys(result.precision))
    sort!(ratings)

    for rating in ratings
        println("Rating $rating:")
        println("Precision - $(result.precision[rating])")
        println("Recall - $(result.recall[rating])")
        println("F1 - $(result.f1[rating])")
        println("")
    end

    println("Global:")
    println("Macro F1 - $(result.macrof1)")
end
