.PHONY=logistic run fucking-around

all:
	exit 0

logistic:
	mkdir -p dist
	ghc logistic.hs -odir dist/ -hidir dist/ -o dist/logistic


multi-descent:
	mkdir -p dist
	ghc -O multi-descent -odir dist/ -hidir dist/ -o dist/multi-descent
	time ./dist/multi-descent


run: logistic
	./dist/logistic


fucking-around:
	mkdir -p dist
	ghc fucking-around -odir dist/ -hidir dist/ -o dist/fucking-around
	./dist/fucking-around


clean:
	rm -f dist/*
