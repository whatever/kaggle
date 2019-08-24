-- Logistic regression implemented in Haskell
-- for fun... I guess


-- Return logistic(x)
logistic x = 1.0 / (1.0 - (exp (-x)))



-- Return dot product between x and y
-- XXX: Reduce memory overhead here /shrug
dot x y = sum [(x!!i)*(y!!i) | i <- [0..(length x)-1]]


-- Return linear function for theta
linearFunction theta = (\x -> (dot theta x))


-- Affine linear function
affineLinearFunction theta = (\x -> (head theta) + (dot (tail theta) x))


-- f(x) = 3.0 x1 + 1.2 x2 + 10 x3
f = (linearFunction [3.0, 1.2, 10.0])


-- g(x) = 2.0 + 2.0 x1
g = (affineLinearFunction [2.0, 2.0])


-- Metric used to compute whether f(x) predicts y
cost f xs ys = (sum [((f x)-y)^2 | (x, y) <- (zip xs ys)]) / (fromIntegral (length xs)) / 2.0


-- X: input vector
xs = [[x, 2, 3] | x <- [1..10]]


-- Y: output vector
ys = [1 | y <- [1..10]]


main = do
  print (logistic 1)
  print (logistic 2.1)
  print (f [1, 1, 1])
  print (dot [0, 2, 3] [1, 2, 3])
  print ((cost g xs ys) / 2.0 / 2.0)
