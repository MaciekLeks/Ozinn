using OzinnFFN

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

train!(model, X, label, maxiters=10000, stepsize=0.01, clipval=1., ϵ=0.1, pull=1.)

#test
println("Test:")

predict(model, X[1,:])
predict(model, X[2,:])
predict(model, X[3,:])
predict(model, X[4,:])
predict(model, X[5,:])
predict(model, X[6,:])
predict(model, X[7,:])
predict(model, X[8,:])
predict(model, X[9,:])
predict(model, X[10,:])
