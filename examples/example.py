from argparse import ArgumentParser
from base64 import b64encode
from io import BytesIO
from pathlib import Path

from PIL import Image

from aesthetic_predictor import predict_aesthetic


def main():
    current_dir = Path(__file__).parent
    parser = ArgumentParser()
    parser.add_argument("--root-dir", type=Path, default=current_dir / "cc0")
    parser.add_argument("--output", type=Path, default=current_dir / "scores.html")
    args = parser.parse_args()
    root_dir: Path = args.root_dir

    files = list(root_dir.glob("**/*.jpg"))
    images = []
    b64_images = []

    for file in files:
        images.append(Image.open(file))
        b64_images.append(b64encode(file.read_bytes()).decode("utf-8"))
    scores = predict_aesthetic(images).numpy().ravel()

    with open(args.output, "w+") as fp:
        row_format = """\
<tr>
    <th scope="row">{number}</th>
    <td>{path}</td>
    <td>{score}</td>
    <td><img width="300px" src="data:image/jpeg;base64,{image}" /></td>
</tr>"""
        results = []
        for i in range(len(b64_images)):
            file = files[i].relative_to(Path().absolute()).as_posix()
            results.append(
                row_format.format(
                    number=i,
                    path=file,
                    score=scores[i],
                    image=b64_images[i],
                )
            )

        fp.write(
            (current_dir / "scores.template.html")
            .read_text()
            .format(table_body="\n".join(results))
        )


if __name__ == "__main__":
    main()
