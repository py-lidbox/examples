HTML_OUT_DIR := ~/code/py-lidbox.github.io
EXAMPLE_DIRS := common-voice-small common-voice-augmenting common-voice-embeddings common-voice-angular-lstm

.PHONY: $(EXAMPLE_DIRS) index

all: index $(EXAMPLE_DIRS)

$(EXAMPLE_DIRS):
	@mkdir -pv $(HTML_OUT_DIR)/$@
	@jupyter nbconvert --to html $@/main.ipynb --stdout > $(HTML_OUT_DIR)/$@/main.html
	@echo "wrote '$(HTML_OUT_DIR)/$@/main.html'"

index:
	@python3 md2html.py $(HTML_OUT_DIR)/index.html
	@echo "wrote '$(HTML_OUT_DIR)/index.html'"
