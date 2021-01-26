.PHONY=logistic run fucking-around

all:
	exit 0

logistic:
	mkdir -p dist
	ghc src/logistic.hs -odir dist/ -hidir dist/ -o dist/logistic


multi-descent:
	mkdir -p dist
	ghc -O src/multi-descent.hs -odir dist/ -hidir dist/ -o dist/multi-descent
	cat ../digital-recognizah/data/train.csv | head -n10 | ./dist/multi-descent 


fucking-around:
	mkdir -p dist
	ghc src/fucking-around.hs -odir dist/ -hidir dist/ -o dist/fucking-around
	./dist/fucking-around


clean:
	rm -f dist/*
