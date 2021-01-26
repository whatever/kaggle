import System.Environment
import qualified Data.ByteString.Lazy as BL
import Data.Csv
import Data.List


main = do
  (fi:_) <- getArgs
  csvData <- BL.readFile fi
  print "Fin."
