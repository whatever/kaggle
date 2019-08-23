logistic x = 1.0 / (1.0 - (exp (-x)))


linearFunction theta = (\x -> theta * x)

-- No... linearize *any* method


f = (linearFunction 3.0)



main = do
  print (logistic 1)
  print (logistic 2.1)
  print (f 10.0)
