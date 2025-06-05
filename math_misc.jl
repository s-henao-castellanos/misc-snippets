"`center_mod(x,b=1)`: modulus operation, but centered around zero. Result goes from -b/2 to b/2."
center_mod(x,b=1) = mod(x+b/2,b) - b/2

"`expit(x)`: logistic sigmoid 1/(1+exp(-x))"
expit(x) = 1/(1+exp(-x))


"`S1_distance(x,y,b=1)`: distance on the topological circle S¹ between angles x and y. Angles range from 0 to b=1 by default."
S1_distance(x,y,b=1) = abs(center_mod(x-y,b))

"Peak to peak of array"
function ptp(x)
    a,b = extrema(x)
    return b-a
end

"mean of the maximum and the minimum"
function midamplitude(arr)
    low,hi = extrema(arr)
    return (hi+low)/2
end

"return the indices of the kth minima and kth maxima"
function argextrema(arr;k=1) 
    return [partialsortperm(arr,1:k); partialsortperm(arr,1:k,rev=true)]
end

"deletes those elements of to_chop which lie on the positions of the k extrema points of the template array. The two can be the same array."
function chop_extrema(to_chop,template;k=1)
    return to_chop[setdiff(eachindex(template),argextrema(template,k=k))]
end



# Incomplete Beta Regularized gradient
import SpecialFunctions
Hyp1(z,a,b) = pFq((a,a,1-b),(a+1,a+1),z)
Hyp2(z,a,b) = pFq((1-a,b,b),(b+1,b+1),z)
incomplete_beta_regularized(z,a,b) = SpecialFunctions.beta_inc(a,b,z)[1]
function incomplete_beta_regularized_gradient(z,a,b)
    r = [z^(a-1)*(1-z)^(b-1), -z^a/a^2*Hyp1(z,a,b), (1-z)^b/b^2 * Hyp2(1-z,a,b)] ./ beta(a,b)
    r[2] += incomplete_beta_regularized(z,a,b)*(log(z)+digamma(a+b)-digamma(a))
    r[3] -= incomplete_beta_regularized(1-z,b,a)*(log(1-z)+digamma(a+b)-digamma(b))
    return r 
end