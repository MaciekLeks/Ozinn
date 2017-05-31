module OzinnRNN

importall Ozinn

export OziRNN, forward, backward, update, cleargrad, train!, predict

type OziRNNLayer{T} <: AbstractOziLayer
  wxh::OziWire{T}
  whh::OziWire{T}
  whb::OziWire{T}
end

OziRNNLayer(T, insize::Int, outsize::Int) =  begin
  OziRNNLayer{T}(OziWire(T, outsize, insize), OziWire(T, outsize, outsize), OziWire(T,outsize))
end

type OziRNN{T} <: AbstractOziModel
  net::OziNet
  xin::Matrix{T}
end



function OziRNN(T, insize, hdsizes::Vector{Int}, outsize, initϵ = 1.) #np. 3,4,5,2 (3in, 4 i 5 to rozmary hidden, i 2 to out) a xref to referencja do wektora wejściowego, który będzie podlegać zmianie
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
    @show whh = OziWire(T, hdsizes[l], hdsizes[l], initϵ, true, true)
    @show whb = OziWire(T, hdsizes[l], initϵ, true, false)

    ht0 = mul(_net, wxh, input)
    ht1 = mul(_net, whh, hdprev) #todo hdprev

    relu(_net, add(_net, add(_net, ht0, ht1), whb); out=hdprev)
  end
  whd = OziWire(T, outsize, hdsizes[end], initϵ, true, true)
  wbd = OziWire(T, outsize, initϵ, true, false)
  add(_net, mul(_net, whd, hdprev), wbd)

  return OziRNN{T}(_net, x.vals)
end

forward{T}(model::OziRNN{T}) = begin
  output = forward(model.net)
end

backward{T}(model::OziRNN{T}) = begin
  backward(model.net)
end

update{T}(model::OziRNN{T}, stepsize::T, clipval::T) = begin
  update(model.net, stepsize, clipval)
end

cleargrad{T}(model::OziRNN{T}) = begin
  cleargrad(model.net)
end

train!{T}(model::OziRNN{T}, X::AbstractArray, label::AbstractArray; maxiters::Signed=10, stepsize::AbstractFloat=0.01, clipval::AbstractFloat=1., ϵ::AbstractFloat=0.1, pull::AbstractFloat=1.) = begin
  train!(model.net, model.xin, X, label, maxiters=maxiters, stepsize=stepsize, clipval=clipval, ϵ=ϵ, pull=pull)
end

predict{T}(model::OziRNN{T}, testval::AbstractArray) = begin
  predict(model.net, model.xin, testval)
end

end #module
