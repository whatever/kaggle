.PHONY=run digital-recognizah clean

digital-recognizah:
	mkdir -p dist
	ghc -O src/lol.hs -odir dist/ -hidir dist/ -o dist/digital-recognizah

run: clean digital-recognizah ./data/train.csv
	./dist/digital-recognizah ./data/train.csv

clean:
	rm -f dist/*
