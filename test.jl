using Test
include("autodiff.jl")  # Wczytanie pliku z definicją AutoDiff
using .AutoDiff  # Użycie modułu AutoDiff

@testset "AutoDiff Test for sin(x^2)" begin
    # Definicja zmiennej x
    x = AutoDiff.Variable(5.0, name="x")
    dwa = AutoDiff.Constant(2.0)  # Stała 2.0

    # Tworzenie wyrażenia sin(x^2)
    expr = sin(x ^ dwa)  # sin(x^2)

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

# @testset "AutoDiff Test for sin(x * y) + cos(x - y)" begin
#     # Definicja zmiennych x i y
#     x = AutoDiff.Variable(3.0, name="x")
#     y = AutoDiff.Variable(2.0, name="y")

#     # Tworzenie wyrażenia sin(x * y) + cos(x - y)
#     expr = sin(x * y) + cos(x - y)

#     # Obliczenia w przód
#     order = AutoDiff.topological_sort(expr)
#     AutoDiff.forward!(order)

#     # Sprawdzenie poprawności wartości funkcji
#     expected_value = sin(3.0 * 2.0) + cos(3.0 - 2.0)
#     @test isapprox(expr.output, expected_value, atol=1e-6)

#     # Obliczenia wstecz
#     AutoDiff.backward!(order)

#     # Sprawdzenie poprawności gradientów
#     expected_grad_x = y.output * cos(x.output * y.output) - sin(x.output - y.output)
#     expected_grad_y = x.output * cos(x.output * y.output) + sin(x.output - y.output)
#     @test isapprox(x.gradient, expected_grad_x, atol=1e-6)
#     @test isapprox(y.gradient, expected_grad_y, atol=1e-6)

#     println("Wynik: ", expr.output)
#     println("Gradient x: ", x.gradient)
#     println("Gradient y: ", y.gradient)
# end

@testset "AutoDiff Test for A * B (Matrix Multiplication)" begin
    # Definicja zmiennych macierzowych
    A = AutoDiff.Variable([1.0 2.0 3.0; 4.0 5.0 6.0], name="A")  # Macierz 2x3
    B = AutoDiff.Variable([7.0 8.0; 9.0 10.0; 11.0 12.0], name="B")  # Macierz 3x2

    # Wyrażenie: A * B (mnożenie macierzowe)
    expr = A * B

    # Ponieważ wynik to macierz, musimy zredukować ją do skalaru (np. przez sumę)
    expr_scalar = sum(expr)

    # Forward pass
    time_sort = @elapsed order = AutoDiff.topological_sort(expr_scalar)
    time_forward = @elapsed AutoDiff.forward!(order)

    # Sprawdzenie poprawności wartości funkcji
    expected_value = sum([1.0 2.0 3.0; 4.0 5.0 6.0] * [7.0 8.0; 9.0 10.0; 11.0 12.0])
    @test isapprox(expr_scalar.output, expected_value, atol=1e-6)

    # Backward pass
    time_backward = @elapsed AutoDiff.backward!(order)

    # Sprawdzenie poprawności gradientów
    # Gradient dla A to ones(2,2) * B', ponieważ d/dA sum(A*B) = ones * B'
    expected_grad_A = ones(2,2) * B.output'
    expected_grad_B = A.output' * ones(2,2)

    @test isapprox(A.gradient, expected_grad_A, atol=1e-6)
    @test isapprox(B.gradient, expected_grad_B, atol=1e-6)

    println("Czas topological_sort: ", time_sort, " s")
    println("Czas forward pass: ", time_forward, " s")
    println("Czas backward pass: ", time_backward, " s")

    println("Wynik (sum(A*B)): ", expr_scalar.output)
    println("Gradient A: ", A.gradient)
    println("Gradient B: ", B.gradient)
end
