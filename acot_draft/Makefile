# ACoT-UAV-Track draft build.
# Usage:
#   make          build main.pdf once
#   make watch    continuous auto-rebuild on save (Ctrl-C to stop)
#   make clean    remove build artifacts (keeps main.pdf)
#   make distclean remove build artifacts AND main.pdf

.PHONY: all watch clean distclean

all:
	./build.sh

watch:
	./watch.sh

clean:
	rm -f main.aux main.log main.out main.bbl main.blg

distclean: clean
	rm -f main.pdf
