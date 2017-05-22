using OzinnFFN

# xin = NnWireEx(Float64,2)
# xin.vals[1] = 1.
# xin.vals[2] = 2.
initϵ = .9
@show model = OziFFN(Float64, 1, [4], 1, initϵ)

 X = [
  0.
  1.
  0.
  1.
  1.
  1.
  0.
  0.
  0.
  1.
 ]
label = [0. 1. 0. 1. 1. 1. 0. 0. 0. 1.]

#{old code to be redefined
maxiters = 10000
stepsize = 0.01
clipval = 1. #wartość przycinania gradientu
ϵ = 0.1 #error
pull = 1.
i = 0
#old code to be redefined}
@time for iter=1:maxiters
  i = i + 1
  j = i % size(X,1) + 1
  model.xin[:] = X[j,:]

  @show out = forward(model)
  score = out.vals

  if all(abs(label .- score) .<= ϵ)
    println("final score = $score after $(iter) iters")
    break
  end

  fill!(out.grads, 0.)
  @show abs(label[j] .- score)
  out.grads[(score .< label[j]) & (abs(label[j] .- score) .> ϵ)] = +pull #- rand(1)[1] #pull up
  out.grads[(score .> label[j]) & (abs(label[j] .- score) .> ϵ)] = -pull #+ rand(1)[1] #pull down

  #out.grads[1] = pull
  backward(model)

  update(model, stepsize, clipval)

  # _w.vals += stepsize .* (_w.grads - _w.vals) #todo regularization on w_1
  # _b.vals += stepsize .* _b.grads

  #println("pull=$pull score = $score _a=$(_a), _b=$(_b), c=$(_c): _g4=$_g4 _g3=$_g3 _g2=$_g2 _g1=_$_g1")
  println("pull=$(out.grads) for out.grads=$(out.grads) score=$score label=$(label[j])")
  #sleep(2)


  cleargrad(model)
end

#test
println("Test:")
model.xin[:] = X[1,:]
@show out = forward(model)
model.xin[:] = X[2,:]
@show out = forward(model)
model.xin[:] = X[3,:]
@show out = forward(model)
model.xin[:] = X[4,:]
@show out = forward(model)
model.xin[:] = X[5,:]
@show out = forward(model)
model.xin[:] = X[6,:]
@show out = forward(model)
model.xin[:] = X[7,:]
@show out = forward(model)
model.xin[:] = X[8,:]
@show out = forward(model)
model.xin[:] = X[9,:]
@show out = forward(model)
model.xin[:] = X[10,:]
@show out = forward(model)
# model.xin[:] = X[4,:]
# @show out = forward(model)
# model.xin[:] = X[5,:]
# @show out = forward(model)
