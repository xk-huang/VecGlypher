from svgpathtools import svg2paths, svgstr2paths, wsvg
from svgpathtools.parser import parse_transform
from svgpathtools.path import transform
from svgpathtools.paths2svg import big_bounding_box


def main():
    svg_path = "misc/data.svg"

    paths, attrs, svg_attr = svg2paths(svg_path, return_svg_attributes=True)

    print(svg_attr)
    print(f"original viewBox: {svg_attr['viewBox']}")

    new_view_box = get_new_bounding_box(paths, attrs)
    wsvg(
        paths,
        attributes=attrs,
        viewbox=new_view_box,
        # svg_attributes=svg_attr,
        filename="misc/data_out.svg",
    )


def get_new_bounding_box(paths, attrs):
    new_paths = []
    for p, a in zip(paths, attrs):
        new_paths.append(transform(p, parse_transform(a["transform"])))

    xmin, xmax, ymin, ymax = big_bounding_box(new_paths)
    dx = xmax - xmin
    dy = ymax - ymin
    return xmin, ymin, dx, dy


if __name__ == "__main__":
    main()
