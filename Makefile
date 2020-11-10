HTML_OUT_DIR := ~/code/py-lidbox.github.io

.PHONY: common-voice-small common-voice-large common-voice-augmenting

common-voice-small common-voice-large common-voice-augmenting:
	@mkdir -pv $(HTML_OUT_DIR)/$@
	@jupyter nbconvert --to html $@/main.ipynb --stdout > $(HTML_OUT_DIR)/$@/main.html
