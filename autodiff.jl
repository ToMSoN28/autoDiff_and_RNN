# autodiff.jl

module AutoDiff

using LinearAlgebra

abstract type GraphNode end
abstract type Operator <: GraphNode end

struct Constant{T} <: GraphNode
    output :: T
end

mutable struct Variable{T} <: GraphNode
    output :: T
    gradient :: Union{T, Nothing}
    name :: String
    # Poprawiono konstruktor, aby Julia wiedziała, że `T` jest parametrem typu
    Variable(output::T; name::String="?") where {T} = new{T}(output, nothing, name)  # ZMIANA
end

mutable struct ScalarOperator{F, T} <: Operator
    inputs :: Vector{GraphNode}
    output :: Union{T, Nothing}
    gradient :: Union{T, Nothing}
    name :: String
    ScalarOperator(fun::F, inputs::GraphNode...; name::String="?") where {F} = begin
        T = promote_type(map(x -> begin
            if x isa Variable
                typeof(x.output)
            elseif x isa Constant
                typeof(x.output)
            elseif x isa Operator
                # Domyślnie Float64, albo nawet Any – lepiej niż Nothing
                Float64
            else
                error("Unknown node type")
            end
        end, inputs)...)
        new{F, T}(collect(inputs), nothing, nothing, name)
    end
end

mutable struct BroadcastedOperator{F, T} <: Operator
    inputs :: Vector{GraphNode}
    output :: Union{Vector{T}, Nothing}
    gradient :: Union{Vector{T}, Nothing}
    name :: String
    BroadcastedOperator(fun::F, inputs::GraphNode...; name::String="?") where {F} = 
        new{F, T}(collect(inputs), nothing, nothing, name)
end

function visit(node::GraphNode, visited::Set{GraphNode}, order::Vector{GraphNode})
    if node ∉ visited
        push!(visited, node)
        push!(order, node)
    end
    return nothing
end

function visit(node::Operator, visited::Set{GraphNode}, order::Vector{GraphNode})
    if node ∉ visited
        push!(visited, node)
        for input in node.inputs
            visit(input, visited, order)
        end
        push!(order, node)
    end
    return nothing
end

function topological_sort(head::GraphNode)
    visited = Set{GraphNode}()
    order = Vector{GraphNode}()
    visit(head, visited, order)
    return order
end

reset!(node::Constant) = nothing
reset!(node::Variable) = node.gradient = nothing
reset!(node::Operator) = node.gradient = nothing

compute!(node::Constant) = nothing
compute!(node::Variable) = nothing
compute!(node::Operator) =
    node.output = forward(node, [input.output for input in node.inputs]...)

function forward!(order::Vector)
    for node in order
        compute!(node)
        reset!(node)
    end
    return last(order).output
end

update!(node::Constant, gradient) = nothing
update!(node::GraphNode, gradient) = if isnothing(node.gradient)
    node.gradient = gradient
else
    node.gradient = node.gradient + gradient
end

function backward!(order::Vector; seed=1.0)
    result = last(order)
    result.gradient = seed
    @assert length(result.output) == 1 "Gradient is defined only for scalar functions"
    for node in reverse(order)
        backward!(node)
    end
    return nothing
end

function backward!(node::Constant) end
function backward!(node::Variable) end
function backward!(node::Operator)
    inputs = node.inputs
    gradients = backward(node, [input.output for input in inputs]..., node.gradient)
    for (input, gradient) in zip(inputs, gradients)
        update!(input, gradient)
    end
    return nothing
end

# -----------------------------------ScalarOperator--------------------------------

# Potęgowanie (^) dla ScalarOperator
import Base: ^
^(x::GraphNode, n::GraphNode) = ScalarOperator(^, x, n)
forward(::ScalarOperator{typeof(^)}, x, n) = x^n
backward(::ScalarOperator{typeof(^)}, x, n, g) = (g * n * x^(n-1), g * log(abs(x)) * x^n)

# Sinus dla ScalarOperator
import Base: sin
sin(x::GraphNode) = ScalarOperator(sin, x)
forward(::ScalarOperator{typeof(sin)}, x) = sin(x)
backward(::ScalarOperator{typeof(sin)}, x, g) = (g * cos(x),)

# Cosinus dla ScalarOperator
import Base: cos
cos(x::GraphNode) = ScalarOperator(cos, x)
forward(::ScalarOperator{typeof(cos)}, x) = cos(x)
backward(::ScalarOperator{typeof(cos)}, x, g) = (-g * sin(x),)

# Tangens hiperboliczny (tanh) dla ScalarOperator
import Base: tanh
tanh(x::GraphNode) = ScalarOperator(tanh, x)
forward(::ScalarOperator{typeof(tanh)}, x) = tanh(x)
backward(::ScalarOperator{typeof(tanh)}, x, g) = (g * (1.0 - tanh(x)^2),)

# Mnożenie dla ScalarOperator
import Base: *
*(x::GraphNode, y::GraphNode) = ScalarOperator(*, x, y)
forward(::ScalarOperator{typeof(*)}, x, y) = x * y
backward(::ScalarOperator{typeof(*)}, x, y, g) = (g * y, g * x)

end # module AutoDiff