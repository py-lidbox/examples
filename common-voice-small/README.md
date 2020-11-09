# Simple spoken language identification

This is a simple example on training a neural network classifier for recognizing languages from speech data.
The data used in this example is from 4 [Mozilla Common Voice](https://commonvoice.mozilla.org/en/datasets) datasets.
You can either download the datasets and run the [example notebook](./main.ipynb), or just read through the notebook as a tutorial.

## Datasets

* Estonian (et)
* Mongolian (mn)
* Tamil (ta)
* Turkish (tr)

These datasets were chosen for this example because they do not contain too much data for a simple example (approx 10 hours each), yet there should be enough data for applying deep learning.

## Prerequisites

`python3` and all packages listed in `requirements.txt`.

TensorFlow is not included, since you might already have a working setup with a GPU and don't want to install a CPU version alongside that one.
If you do not have TensorFlow and do not want to configure GPUs etc, try:
```
pip install tensorflow
```

### Getting the data

Download the 4 datasets listed above from the [Common Voice](https://commonvoice.mozilla.org/en/datasets) website (you need to enter your email) and put them somewhere into a single directory.
In this example that directory is `./downloads`.

After downloading all archives, extract all into a single directory `cv-corpus` (with `bash`):
```bash
mkdir -pv ./cv-corpus
for archive in et mn ta tr; do
    tar zx -C ./cv-corpus -f ./downloads/${archive}.tar.gz --strip-components 1
done
```

The extracted directory should now look like this (`clips` is a dir with a lot of mp3 files):
```
tree -L 2 ./cv-corpus
./cv-corpus
├── et
│   ├── clips
│   ├── dev.tsv
│   ├── invalidated.tsv
│   ├── other.tsv
│   ├── reported.tsv
│   ├── test.tsv
│   ├── train.tsv
│   └── validated.tsv
├── mn
│   ├── clips
│   ├── dev.tsv
│   ├── invalidated.tsv
│   ├── other.tsv
│   ├── reported.tsv
│   ├── test.tsv
│   ├── train.tsv
│   └── validated.tsv
├── ta
│   ├── clips
│   ├── dev.tsv
│   ├── invalidated.tsv
│   ├── other.tsv
│   ├── reported.tsv
│   ├── test.tsv
│   ├── train.tsv
│   └── validated.tsv
└── tr
    ├── clips
    ├── dev.tsv
    ├── invalidated.tsv
    ├── other.tsv
    ├── reported.tsv
    ├── test.tsv
    ├── train.tsv
    └── validated.tsv

8 directories, 28 files
```

## Running

```
jupyter notebook main.ipynb
```

## Remove caches to free up space

```
rm -rv cache
```
