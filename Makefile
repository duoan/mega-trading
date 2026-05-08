.PHONY: test demo

test:
	python3 -m unittest discover -s tests

demo:
	python3 -m marketfm --help
