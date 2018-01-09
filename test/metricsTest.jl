@testset "Decision Metrics Test" begin
    @testset "Integer Preferences" begin
        preferences = Persa.RatingPreferences([1, 2, 3, 4, 5])
        threshold = 4.0

        labels = []
        predicts = []

        @test_throws AssertionError Persa.DecisionMetrics(labels, predicts, preferences, threshold)

        labels = [1, 2]
        predicts = [1]

        @test_throws AssertionError Persa.DecisionMetrics(labels, predicts, preferences, threshold)

        labels = [1]
        predicts = [1, 2]

        @test_throws AssertionError Persa.DecisionMetrics(labels, predicts, preferences, threshold)

        labels = [1, 2, 3, 4, 5]
        predicts = [1, 2, 3, 4, 5]

        @test_throws AssertionError Persa.DecisionMetrics(labels, predicts, preferences, 10)
        @test_throws AssertionError Persa.DecisionMetrics(labels, predicts, preferences, -1)

        result = Persa.DecisionMetrics(labels, predicts, preferences, threshold)

        @test Persa.precision(result) == 1.00
        @test Persa.recall(result) == 1.00
        @test Persa.f1score(result) == 1.00

        for i in unique(preferences)
            @test Persa.precision(result, i) == 1.00
            @test Persa.recall(result, i) == 1.00
            @test Persa.f1score(result, i) == 1.00
        end

        labels = [1, 1, 2, 3, 4, 5]
        predicts = [1, 2, 2, 3, 4, 5]

        result = Persa.DecisionMetrics(labels, predicts, preferences, threshold)

        @test Persa.precision(result) == 1.00
        @test Persa.recall(result) == 1.00
        @test Persa.f1score(result) == 1.00

        i = 1
        @test Persa.precision(result, i) == 1.00
        @test Persa.recall(result, i) == 0.5
        @test Persa.f1score(result, i) == 2/3

        i = 2
        @test Persa.precision(result, i) == 0.5
        @test Persa.recall(result, i) == 1.0
        @test Persa.f1score(result, i) == 2/3

        for i=3:5
            @test Persa.precision(result, i) == 1.0
            @test Persa.recall(result, i) == 1.0
            @test Persa.f1score(result, i) == 1.0
        end

        labels = [1, 2, 3, 4, 5]
        predicts = [2, 1, 4, 5, 4]

        result = Persa.DecisionMetrics(labels, predicts, preferences, threshold)

        for i in unique(preferences)
            @test Persa.precision(result, i) == 0.0
            @test Persa.recall(result, i) == 0.0
            @test Persa.f1score(result, i) == 0.0
        end

        labels = [1, 2, 3, 4, 5]
        predicts = [1, 2, 3, 4, 1]

        result = Persa.DecisionMetrics(labels, predicts, preferences, threshold)

        @test Persa.precision(result) == 1.0
        @test Persa.recall(result) == 0.5
        @test Persa.f1score(result) == 2/3

        labels = [1, 2, 3, 4, 5]
        predicts = [1, 2, 4, 4, 1]

        result = Persa.DecisionMetrics(labels, predicts, preferences, threshold)

        @test Persa.precision(result) == 0.5
        @test Persa.recall(result) == 0.5
        @test Persa.f1score(result) == 0.5

        labels = [1, 2, 3, 4, 5]
        predicts = [1, 2, 3, 1, 1]

        result = Persa.DecisionMetrics(labels, predicts, preferences, threshold)

        @test Persa.precision(result) == 0.0
        @test Persa.recall(result) == 0.0
        @test Persa.f1score(result) == 0.0

        labels = [1, 2, 3, 4, 5]
        predicts = [1, 4, 4, 4, 5]

        result = Persa.DecisionMetrics(labels, predicts, preferences, threshold)

        @test Persa.precision(result) == 0.5
        @test Persa.recall(result) == 1.0
        @test Persa.f1score(result) == 2/3
    end
end
