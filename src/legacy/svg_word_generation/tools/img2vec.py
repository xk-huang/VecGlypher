from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument("input", help="Input image file.")
parser.add_argument("output", help="Output SVG file.")
parser.add_argument("--backend", default="vtracer", help="Backend to use.")


args = parser.parse_args()
input_path = args.input
output_path = args.output
backend = args.backend

if backend == "vtracer":
    import vtracer

    vtracer.convert_image_to_svg_py(input_path, output_path)
elif backend == "autotrace":
    import numpy as np
    from autotrace import Bitmap, VectorFormat
    from PIL import Image

    image = np.asarray(Image.open(input_path).convert("RGB"))
    # Create a bitmap.
    bitmap = Bitmap(image)
    # Trace the bitmap.
    vector = bitmap.trace()

    # Save the vector as an SVG.
    vector.save(output_path)
    # Get an SVG as a byte string.
    # svg = vector.encode(VectorFormat.SVG)
else:
    raise ValueError(f"Unknown backend: {backend}")
