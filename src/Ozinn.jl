module Ozinn

export AbstractOziWire, AbstractOziLayer, AbstractOziModel, OziWire, OziNet
export add, mul, sigmoid, relu, tanh, forward, backward, cleargrad, update

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


#mini warstwa - tj. warstwa ale na poziomie pojedynczej operacji mul, add, tanh, sigmoid,...
type OziCoating
  out::AbstractOziWire
  inw::Vector{AbstractOziWire}
  forward::Function
  backward::Function
  OziCoating(out::AbstractOziWire, inws::AbstractOziWire...) = new(out, [inws...])
end

#std recurrent nn

#powinien być w środku modelu
type OziNet
  coatings::Vector{OziCoating}
  # tagged::Vector{NnCoating} #layer decoders
  OziNet() = new(Array(OziCoating,0))
end


function add{T,N,D}(net::OziNet, inws::OziWire{T,N,D}...)
  out = OziWire(T, N, D)
  coating = OziCoating(out, inws...)
  coating.forward = () -> begin
    fill!(out.vals, zero(T))
    @inbounds for inw in inws
      @inbounds for j=1:D, i=1:N #todo: czy lepiej iść po kolei wiersze góra dół i potem następna kolumna, czy kolumny i dopiero wiersze
        out.vals[i,j] += inw.vals[i,j]
      end
    end
    return out
  end
  coating.backward = () -> begin
        @inbounds for inw in inws
          @inbounds for j=1:D, i=1:N #todo: czy lepiej iść po kolei wiersze góra dół i potem następna kolumna, czy kolumny i dopiero wiersze
            inw.grads[i,j] += out.grads[i,j] #1.0 * dout
          end
        #end
        end
  end
  push!(net.coatings, coating)
  return out
end


function mul{T, N, D, M}(net::OziNet, inw1::OziWire{T,N,D}, inw2::OziWire{T,D,M})
  #@show N, D, M
  out = OziWire(T, N, M) # (N,D)*(D,M) > (N,M)
  coating = OziCoating(out, inw1, inw2)
  coating.forward = () -> begin
    @inbounds out.vals = A_mul_B!(out.vals, inw1.vals, inw2.vals) #przeliczenie
    return out
  end
  coating.backward = () ->
    begin
      @inbounds for i=1:N,j=1:M
        dout = out.grads[i,j]
        @inbounds for k=1:D
          inw1.grads[i,k] = inw2.vals[k,j] * dout
          inw2.grads[k,j] = inw1.vals[i,k] * dout
        end
      end
    end
  push!(net.coatings, coating)
  return out
end

function tanh{T, N, D}(net::OziNet, inw::OziWire{T,N,D})
  out = OziWire(T, N, D)
  coating = OziCoating(out, inw)
  coating.forward = () -> begin
      #@inbounds out.vals = A_mul_B!(out.vals, in1.vals, in2.vals) #przeliczenie
    @inbounds for j=1:D, i=1:N #todo: czy lepiej iść po kolei wiersze góra dół i potem następna kolumna, czy kolumny i dopiero wiersze
      out.vals[i,j] = Base.tanh(inw.vals[i,j])
    end
    return out
  end
  coating.backward = () -> begin
      ident = one(T)
      @inbounds for j=1:D, i=1:N #todo: czy lepiej iść po kolei wiersze góra dół i potem następna kolumna, czy kolumny i dopiero wiersze
        inw.grads[i,j] = (ident - Base.tanh(inw.vals[i,j])^2) * out.grads[i,j]
      end
    end
  push!(net.coatings, coating)
  return out
end


function sigmoid{T, N, D}(net::OziNet, inw::OziWire{T,N,D}; out=nothing)
  if out == nothing
    out = OziWire(T, N, D)
  end
  coating = OziCoating(out, inw)
  ident = one(T) #e.g 1.0
  coating.forward = () -> begin
    @inbounds for j=1:D, i=1:N #todo: czy lepiej iść po kolei wiersze góra dół i potem następna kolumna, czy kolumny i dopiero wiersze
      out.vals[i,j] = ident / (ident + exp(-inw.vals[i,j]))
    end
    return out
  end
  coating.backward = () -> begin
      @inbounds for j=1:D, i=1:N #todo: czy lepiej iść po kolei wiersze góra dół i potem następna kolumna, czy kolumny i dopiero wiersze
        inw.grads[i,j] = out.vals[i,j] * (ident - out.vals[i,j]) * out.grads[i,j]
      end
    end
  push!(net.coatings, coating)
  return out
end


###
# out - predefined matrix
###
function relu{T, N, D}(net::OziNet, inw::OziWire{T,N,D}; out=nothing)
  if out == nothing
    out = OziWire(T, N, D)
  end
  coating = OziCoating(out, inw)
  ident1 = one(T) #e.g 1.0
  ident0 = zero(T) #e.g. 0.0
  coating.forward = () -> begin
    @inbounds for j=1:D, i=1:N #todo: czy lepiej iść po kolei wiersze góra dół i potem następna kolumna, czy kolumny i dopiero wiersze
      out.vals[i,j] = max(inw.vals[i,j], ident0)
    end
    return out
  end
  coating.backward = () -> begin
      @inbounds for j=1:D, i=1:N #todo: czy lepiej iść po kolei wiersze góra dół i potem następna kolumna, czy kolumny i dopiero wiersze
        inw.grads[i,j] = inw.vals[i,j] > ident0 ? ident1 * out.vals[i,j] : ident0
      end
    end
  push!(net.coatings, coating)
  # if (tagged) #layer decoder
  #   push!(net.tagged, coating)
  # end
  return out
end


function forward(net::OziNet)
  res = nothing
  @inbounds for i=1:length(net.coatings)
    res = net.coatings[i].forward()
    #println("forward $res")
  end
  return res
end



function backward(net::OziNet)
  @inbounds for i=length(net.coatings):-1:1
    net.coatings[i].backward()
  end
end

function update{T}(net::OziNet, stepsize::T, clipval::T)
  #print("updateEx: coatings:$(length(net.coatings)):")
  @inbounds for i=length(net.coatings):-1:1
    inputs = net.coatings[i].inw
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
  len = length(net.coatings)
  if (len) < 1
    return
  end

  tp = eltype(net.coatings[1].out.grads)
  zr = zero(tp)

  @inbounds for i=len:-1:1
    #fill!(_net.cells[i].in1.vals, zr)
    inw = net.coatings[i].inw
    @inbounds for k=1:length(inw)
      fill!(inw[k].grads, zr)
    end
    #fill!(_net.cells[i].in2.vals, zr)
    #fill!(_net.cells[i].in2.grads, zr)
    #fill!(_net.cells[i].out.vals, zr)
    fill!(net.coatings[i].out.grads, zr)
  end
end


end #module
