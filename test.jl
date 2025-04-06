using Test
include("autodiff.jl")  # Wczytanie pliku z definicją AutoDiff
using .AutoDiff  # Użycie modułu AutoDiff

@testset "AutoDiff Test for sin(x^2)" begin
    # Definicja zmiennej x
    x = AutoDiff.Variable(5.0, name="x")

    # Tworzenie wyrażenia sin(x^2)
    expr = sin(x * x)  # sin(x^2)

    # Obliczenia w przód
    order = AutoDiff.topological_sort(expr)
    AutoDiff.forward!(order)

    # Sprawdzenie poprawności wartości funkcji
    @test isapprox(expr.output, sin(5.0^2), atol=1e-6)

    # Obliczenia wstecz
    AutoDiff.backward!(order)

    # Sprawdzenie poprawności gradientu
    expected_grad = 2 * 5.0 * cos(5.0^2)
    @test isapprox(x.gradient, expected_grad, atol=1e-6)

    println("Wynik: ", expr.output)
    println("Gradient: ", x.gradient)
end

@testset "AutoDiff Test for sin(x * y) + cos(x - y)" begin
    # Definicja zmiennych x i y
    x = AutoDiff.Variable(3.0, name="x")
    y = AutoDiff.Variable(2.0, name="y")

    # Tworzenie wyrażenia sin(x * y) + cos(x - y)
    expr = sin(x * y) + cos(x - y)

    # Obliczenia w przód
    order = AutoDiff.topological_sort(expr)
    AutoDiff.forward!(order)

    # Sprawdzenie poprawności wartości funkcji
    expected_value = sin(3.0 * 2.0) + cos(3.0 - 2.0)
    @test isapprox(expr.output, expected_value, atol=1e-6)

    # Obliczenia wstecz
    AutoDiff.backward!(order)

    # Sprawdzenie poprawności gradientów
    expected_grad_x = y.output * cos(x.output * y.output) - sin(x.output - y.output)
    expected_grad_y = x.output * cos(x.output * y.output) + sin(x.output - y.output)
    @test isapprox(x.gradient, expected_grad_x, atol=1e-6)
    @test isapprox(y.gradient, expected_grad_y, atol=1e-6)

    println("Wynik: ", expr.output)
    println("Gradient x: ", x.gradient)
    println("Gradient y: ", y.gradient)
end

@testset "AutoDiff Test for exp.(A * B)" begin
    # Definicja zmiennych macierzowych
    A = AutoDiff.Variable([1.0 2.0; 3.0 4.0], name="A")  # Macierz 2x2
    B = AutoDiff.Variable([0.5 1.5; 2.5 3.5], name="B")  # Macierz 2x2

    # Tworzenie wyrażenia: exp.(A * B) (broadcastowany exparytm na wyniku mnożenia macierzy)
    expr = exp.(A * B)
    y = sum(expr)  # Suma elementów macierzy

    # Obliczenia w przód
    order = AutoDiff.topological_sort(y)
    ptint("order: ", order)
    AutoDiff.forward!(order)

    # Sprawdzenie poprawności wartości funkcji
    expected_value = sum(exp.([1.0 2.0; 3.0 4.0] * [0.5 1.5; 2.5 3.5]))  # exp.(A * B)
    @test isapprox(expr.output, expected_value, atol=1e-6)
    println("Wynik: ", expr.output)

    # Obliczenia wstecz
    AutoDiff.backward!(order)

    # Sprawdzenie poprawności gradientów
    expected_grad_A = (1.0 ./ ([1.0 2.0; 3.0 4.0] * [0.5 1.5; 2.5 3.5])) * B.output'  # Gradient dla A
    expected_grad_B = A.output' * (1.0 ./ ([1.0 2.0; 3.0 4.0] * [0.5 1.5; 2.5 3.5]))  # Gradient dla B
    @test isapprox(A.gradient, expected_grad_A, atol=1e-6)
    @test isapprox(B.gradient, expected_grad_B, atol=1e-6)

    println("Wynik: ", expr.output)
    println("Gradient A: ", A.gradient)
    println("Gradient B: ", B.gradient)
end