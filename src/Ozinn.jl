module Ozinn

export AbstractOziWire, AbstractOziLayer, AbstractOziModel, OziWire, OziNet
export add, mul, sigmoid, relu, tanh, forward, backward, cleargrad, update, train!, predict

abstract AbstractOziWire
abstract AbstractOziLayer
abstract AbstractOziModel

type OziWire{T,N,D} <: AbstractOziWire
  vals::Matrix{T}
  grads::Matrix{T}
  updatable::Bool #updated after backpropagation phase
  regularized::Bool #regularized
end
OziWire(T, n::Int) = @show OziWire{T,n,1}(zeros(T, n, 1), zeros(T, n, 1), false,false)
OziWire(T, n::Int, initϵ, tobeupdated::Bool=false, toberegularized::Bool=false) = @show OziWire{T,n,1}(rand(T, n, 1) * 2initϵ - initϵ, zeros(T, n, 1), tobeupdated, toberegularized) #todo init powinno mieć typ T, ale to nie zadziała
OziWire(T, n::Int, d::Int) = @show OziWire{T,n,d}(zeros(T, n, d), zeros(T, n, d), false, false)
OziWire(T, n::Int, d::Int, initϵ, tobeupdated::Bool=false, toberegularized::Bool=false) = @show OziWire{T,n,d}(rand(T, n, d) * 2initϵ - initϵ, zeros(T, n, d), tobeupdated, toberegularized) #todo init powinno mieć typ T, ale to nie zadziała

OziWire{T}(vals::Matrix{T}) = @show OziWire{T,size(vals,1),size(vals,2)}(vals, zeros(T, size(vals,1), size(vals,2)))
OziWire{T}(vals::Vector{T}) = @show OziWire{T,size(vals,1),size(vals,2)}(reshape(vals, length(vals),1), zeros(T, size(vals,1), size(vals,2)))
function Base.length{T,N,D}(v::OziWire{T,N,D})
  N
end
function Base.eltype{T,N,D}(::OziWire{T,N,D})
  @show T
end


# micro layer at the level of simple operation, e.g. add, mull, tanh, sigmoid,...
type OziGate
  out::AbstractOziWire
  inw::Vector{AbstractOziWire}
  forward::Function
  backward::Function
  OziGate(out::AbstractOziWire, inws::AbstractOziWire...) = new(out, [inws...])
end

#std recurrent nn

#powinien być w środku modelu
type OziNet
  gates::Vector{OziGate}
  # tagged::Vector{NnGate} #layer decoders
  OziNet() = new(Array(OziGate,0))
end


function add{T,N,D}(net::OziNet, inws::OziWire{T,N,D}...)
  out = OziWire(T, N, D)
  gate = OziGate(out, inws...)
  gate.forward = () -> begin
    fill!(out.vals, zero(T))
    @inbounds for inw in inws
      @inbounds for j=1:D, i=1:N #todo: czy lepiej iść po kolei wiersze góra dół i potem następna kolumna, czy kolumny i dopiero wiersze
        out.vals[i,j] += inw.vals[i,j]
      end
    end
    return out
  end
  gate.backward = () -> begin
        @inbounds for inw in inws
          @inbounds for j=1:D, i=1:N #todo: czy lepiej iść po kolei wiersze góra dół i potem następna kolumna, czy kolumny i dopiero wiersze
            inw.grads[i,j] += out.grads[i,j] #1.0 * dout
          end
        #end
        end
  end
  push!(net.gates, gate)
  return out
end


function mul{T, N, D, M}(net::OziNet, inw1::OziWire{T,N,D}, inw2::OziWire{T,D,M})
  #@show N, D, M
  out = OziWire(T, N, M) # (N,D)*(D,M) > (N,M)
  gate = OziGate(out, inw1, inw2)
  gate.forward = () -> begin
    @inbounds out.vals = A_mul_B!(out.vals, inw1.vals, inw2.vals) #przeliczenie
    return out
  end
  gate.backward = () ->
    begin
      @inbounds for i=1:N,j=1:M
        dout = out.grads[i,j]
        @inbounds for k=1:D
          inw1.grads[i,k] = inw2.vals[k,j] * dout
          inw2.grads[k,j] = inw1.vals[i,k] * dout
        end
      end
    end
  push!(net.gates, gate)
  return out
end

function tanh{T, N, D}(net::OziNet, inw::OziWire{T,N,D})
  out = OziWire(T, N, D)
  gate = OziGate(out, inw)
  gate.forward = () -> begin
      #@inbounds out.vals = A_mul_B!(out.vals, in1.vals, in2.vals) #przeliczenie
    @inbounds for j=1:D, i=1:N #todo: czy lepiej iść po kolei wiersze góra dół i potem następna kolumna, czy kolumny i dopiero wiersze
      out.vals[i,j] = Base.tanh(inw.vals[i,j])
    end
    return out
  end
  gate.backward = () -> begin
      ident = one(T)
      @inbounds for j=1:D, i=1:N #todo: czy lepiej iść po kolei wiersze góra dół i potem następna kolumna, czy kolumny i dopiero wiersze
        inw.grads[i,j] = (ident - Base.tanh(inw.vals[i,j])^2) * out.grads[i,j]
      end
    end
  push!(net.gates, gate)
  return out
end


function sigmoid{T, N, D}(net::OziNet, inw::OziWire{T,N,D}; out=nothing)
  if out == nothing
    out = OziWire(T, N, D)
  end
  gate = OziGate(out, inw)
  ident = one(T) #e.g 1.0
  gate.forward = () -> begin
    @inbounds for j=1:D, i=1:N #todo: czy lepiej iść po kolei wiersze góra dół i potem następna kolumna, czy kolumny i dopiero wiersze
      out.vals[i,j] = ident / (ident + exp(-inw.vals[i,j]))
    end
    return out
  end
  gate.backward = () -> begin
      @inbounds for j=1:D, i=1:N #todo: czy lepiej iść po kolei wiersze góra dół i potem następna kolumna, czy kolumny i dopiero wiersze
        inw.grads[i,j] = out.vals[i,j] * (ident - out.vals[i,j]) * out.grads[i,j]
      end
    end
  push!(net.gates, gate)
  return out
end


###
# out - predefined matrix
###
function relu{T, N, D}(net::OziNet, inw::OziWire{T,N,D}; out=nothing)
  if out == nothing
    out = OziWire(T, N, D)
  end
  gate = OziGate(out, inw)
  ident1 = one(T) #e.g 1.0
  ident0 = zero(T) #e.g. 0.0
  gate.forward = () -> begin
    @inbounds for j=1:D, i=1:N #todo: czy lepiej iść po kolei wiersze góra dół i potem następna kolumna, czy kolumny i dopiero wiersze
      out.vals[i,j] = max(inw.vals[i,j], ident0)
    end
    return out
  end
  gate.backward = () -> begin
      @inbounds for j=1:D, i=1:N #todo: czy lepiej iść po kolei wiersze góra dół i potem następna kolumna, czy kolumny i dopiero wiersze
        inw.grads[i,j] = inw.vals[i,j] > ident0 ? ident1 * out.vals[i,j] : ident0
      end
    end
  push!(net.gates, gate)
  # if (tagged) #layer decoder
  #   push!(net.tagged, gate)
  # end
  return out
end


function forward(net::OziNet)
  res = nothing
  @inbounds for i=1:length(net.gates)
    res = net.gates[i].forward()
    #println("forward $res")
  end
  return res
end



function backward(net::OziNet)
  @inbounds for i=length(net.gates):-1:1
    net.gates[i].backward()
  end
end

function update{T}(net::OziNet, stepsize::T, clipval::T)
  #print("updateEx: gates:$(length(net.gates)):")
  @inbounds for i=length(net.gates):-1:1
    inputs = net.gates[i].inw
    #print("inw[$i]: $(length(inputs)):")
    @inbounds for j=length(inputs):-1:1
      #print("inputs[$(j)]: $(inputs[j])")
      if (inputs[j].updatable)
        pull = inputs[j].grads
        #clipping
        pull[pull .> clipval] = clipval
        pull[pull .< -clipval] = -clipval
        if (inputs[j].regularized)
          pull -= inputs[j].vals
        end
        inputs[j].vals += stepsize .* pull
        println("force: $pull for grads: $(inputs[j].grads)")
      end
    end
  end
end



function cleargrad(net::OziNet)
  len = length(net.gates)
  if (len) < 1
    return
  end

  tp = eltype(net.gates[1].out.grads)
  zr = zero(tp)

  @inbounds for i=len:-1:1
    #fill!(_net.cells[i].in1.vals, zr)
    inw = net.gates[i].inw
    @inbounds for k=1:length(inw)
      fill!(inw[k].grads, zr)
    end
    #fill!(_net.cells[i].in2.vals, zr)
    #fill!(_net.cells[i].in2.grads, zr)
    #fill!(_net.cells[i].out.vals, zr)
    fill!(net.gates[i].out.grads, zr)
  end
end

train!(net::OziNet, xin::AbstractArray, X::AbstractArray, label::AbstractArray; maxiters::Signed=10, stepsize::AbstractFloat=0.01, clipval::AbstractFloat=1., ϵ::AbstractFloat=0.1, pull::AbstractFloat=1.) = begin
  i = 0
  @time for iter=1:maxiters
    i = i + 1
    j = i % size(X,1) + 1
    xin[:] = X[j,:]

    out = forward(net)
    score = out.vals

    if all(abs(label .- score) .<= ϵ)
      println("final score = $score after $(iter) iters")
      break
    end

    fill!(out.grads, 0.)
    #@show abs(label[j] .- score)
    out.grads[(score .< label[j]) & (abs(label[j] .- score) .> ϵ)] = +pull #- rand(1)[1] #pull up
    out.grads[(score .> label[j]) & (abs(label[j] .- score) .> ϵ)] = -pull #+ rand(1)[1] #pull down

    backward(net)

    update(net, stepsize, clipval)

    println("out.grads=$(out.grads) score=$score label=$(label[j])")

    cleargrad(net)
  end
end

function predict(net::OziNet, xin::AbstractArray, testval::AbstractArray)
  xin[:] = testval #ref
  @show out = forward(net)
end


end #module
