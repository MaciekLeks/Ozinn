module OzinnFFN #Feed Forward Network

importall Ozinn

export OziFFN, forward, backward, update, cleargrad, train!, predict

type OziFFNLayer{T} <: AbstractOziLayer
  wxh::OziWire{T}
  whh::OziWire{T}
end

OziFFNLayer(T, insize::Int, outsize::Int) =  begin
  OziFFNLayer{T}(OziWire(T, outsize, insize), OziWire(T, outsize, outsize), OziWire(T,outsize))
end

type OziFFN{T} <: AbstractOziModel
  net::OziNet
  xin::Matrix{T}
end

function OziFFN(T, insize, hdsizes::Vector{Int}, outsize, initϵ = 1.) #np. 3,4,5,2 (3in, 4 i 5 to rozmary hidden, i 2 to out) a xref to referencja do wektora wejściowego, który będzie podlegać zmianie
  _net = OziNet()
  hdprev = nothing #wyjście z ostatniej ukrytej warstwy
  x = OziWire(T,insize)

  #todo:
  # Xref nie powinien być wężem a tylko wektorem - może zwracać wąż?
  for l in 1:length(hdsizes)
    prevsize = l == 1 ? insize : hdsizes[l-1]
    input = l == 1 ? x : hdprev
    hdprev = OziWire(T, hdsizes[l]) #musi być po input

    @show input
    @show wxh = OziWire(T, hdsizes[l], prevsize, initϵ, true, true)

    sigmoid(_net, mul(_net, wxh, input); out=hdprev)
  end

  return OziFFN{T}(_net, x.vals)
end

forward{T}(model::OziFFN{T}) = begin
  output = forward(model.net)
end

backward{T}(model::OziFFN{T}) = begin
  backward(model.net)
end

update{T}(model::OziFFN{T}, stepsize::T, clipval::T) = begin
  update(model.net, stepsize, clipval)
end

cleargrad{T}(model::OziFFN{T}) = begin
  cleargrad(model.net)
end

train!{T}(model::OziFFN{T}, X::AbstractArray, label::AbstractArray; maxiters::Signed=10, stepsize::AbstractFloat=0.01, clipval::AbstractFloat=1., ϵ::AbstractFloat=0.1, pull::AbstractFloat=1.) = begin
  train!(model.net, model.xin, X, label, maxiters=maxiters, stepsize=stepsize, clipval=clipval, ϵ=ϵ, pull=pull)
end

predict{T}(model::OziFFN{T}, testval::AbstractArray) = begin
  predict(model.net, model.xin, testval)
end

end #module
