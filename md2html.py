import argparse
import string

import mistune


def readf(path):
    with open(path) as f:
        return f.read()


def main(outpath, template="./index.template.html", md_body="./index.md"):
    html_body = mistune.markdown(readf(md_body))
    html_full = string.Template(readf(template)).substitute(body=html_body)
    with open(outpath, "w") as out_f:
        print(html_full, file=out_f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("outpath")

    args = parser.parse_args()
    main(args.outpath)
