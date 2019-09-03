.PHONY=logistic run

all:
	exit 0

logistic:
	mkdir -p dist
	ghc logistic.hs -odir dist/ -hidir dist/ -o dist/logistic


multi-descent:
	mkdir -p dist
	ghc multi-descent -odir dist/ -hidir dist/ -o dist/multi-descent
	./dist/multi-descent


run: logistic
	./dist/logistic


clean:
	rm -f dist/*
