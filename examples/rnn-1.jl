using OzinnRNN

initϵ = .9
@show model = OziRNN(Float64, 1, [3, 3], 1, initϵ)

#try to learn the next value for the given one at the input, in:1 -> out:2, in:4 -> out:5, in:8 -> out:9
#  X = [
#   1.
#   2.
#   3.
#   4.
#   5.
#   6.
#   7.
#   8.
#   9.
#   10.
#  ]
# label = [2. 3. 4. 5. 6. 7. 8. 9. 10. 11.]

X = collect(1:100)
label = collect(2:101)'


train!(model, X, label, maxiters=500, stepsize=0.01, clipval=1., ϵ=0.1, pull=1.)

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
