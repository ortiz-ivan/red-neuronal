import os
import urllib.request
import gzip
import shutil

BASE_URL = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"
FILES = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
]

DEST_DIR = "data/fashion-mnist/"


def download_file(filename):
    url = BASE_URL + filename
    dest_path = os.path.join(DEST_DIR, filename)

    print(f"Descargando: {filename}")
    urllib.request.urlretrieve(url, dest_path)
    print("Listo")


def decompress_file(filename):
    gz_path = os.path.join(DEST_DIR, filename)
    out_path = gz_path.replace(".gz", "")

    print(f"Descomprimiendo: {filename}...")
    with gzip.open(gz_path, "rb") as f_in:
        with open(out_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    print("OK")


def main():
    os.makedirs(DEST_DIR, exist_ok=True)

    for fname in FILES:
        download_file(fname)
        decompress_file(fname)

    print("Descarga completa.")
    print(f"Los archivos quedaron en: {DEST_DIR}")


if __name__ == "__main__":
    main()
