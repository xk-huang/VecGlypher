# Copied from: https://github.com/ximinng/LLM4SVG/blob/375d7c479949873ffdaace8aaf019a03d728837b/llm4svg/data/svg_dataset.py#L14
import re
import xml.etree.ElementTree as ET

from lxml import etree

"""Flatten and inherit group"""


def apply_g_attributes_to_children(svg_string):
    root = etree.fromstring(svg_string)

    for g_tag in root.xpath(
        "//svg:g", namespaces={"svg": "http://www.w3.org/2000/svg"}
    ):
        # get <g> parent
        parent = g_tag.getparent()
        # <g> index
        g_index = parent.index(g_tag)
        # get <g> attributes
        g_attributes = g_tag.attrib

        for i, child in enumerate(g_tag):
            # inherit <g> attributes
            for attr, value in g_attributes.items():
                if attr not in child.attrib:
                    child.attrib[attr] = value

            # moves the <g>' child to the location of its parent
            parent.insert(g_index + i, child)

        # delete <g>
        parent.remove(g_tag)

    return etree.tostring(root, encoding="utf-8").decode()


"""Simplify gradient tags"""


def hex_to_rgb(hex_color):
    # Ensure it's a valid hex color, with optional '#'
    hex_pattern = re.compile(r"^#?([A-Fa-f0-9]{6})$")

    match = hex_pattern.match(hex_color)
    if not match:
        raise ValueError(f"Invalid hex color: {hex_color}")

    # Remove the '#' if present
    hex_color = match.group(1)

    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


def rgb_to_hex(rgb):
    return "#" + "".join(f"{c:02x}" for c in rgb)


def average_color(colors):
    if not colors:
        return None

    avg_r = sum(c[0] for c in colors) // len(colors)
    avg_g = sum(c[1] for c in colors) // len(colors)
    avg_b = sum(c[2] for c in colors) // len(colors)

    return rgb_to_hex((avg_r, avg_g, avg_b))


def get_gradient_color(gradient_id, gradients, root, ns):
    """
    find color by gradient_id，
    and Supports access to other gradients via xlink:href
    """
    if gradient_id in gradients:
        return gradients[gradient_id]

    # Find the gradient associated with the gradient id
    gradient = root.xpath(
        f'//svg:radialGradient[@id="{gradient_id}"] | //svg:linearGradient[@id="{gradient_id}"]',
        namespaces=ns,
    )
    if gradient:
        gradient = gradient[0]
        # check xlink:href
        xlink_href = gradient.get("{http://www.w3.org/1999/xlink}href")
        if xlink_href:
            referenced_id = xlink_href[1:]  # remove '#'
            return get_gradient_color(referenced_id, gradients, root, ns)

    return None  # color not found


def get_previous_fill_color(paths, index):
    """Gets the fill color of the last valid path."""
    colors = []
    while index >= 0:
        fill = paths[index].get("fill")
        if fill and not fill.startswith("url(#"):
            try:
                # Attempt to convert the color to RGB to verify it's valid
                hex_to_rgb(fill)
                colors.append(fill)
            except ValueError:
                # Skip invalid hex colors
                pass
        index -= 1
    return colors


def replace_gradient_tags(
    svg_string: str, fill_is_empty: str = "previous"  # 'skip', 'default'
):
    root = etree.fromstring(svg_string)

    ns = {"svg": "http://www.w3.org/2000/svg", "xlink": "http://www.w3.org/1999/xlink"}

    # find all gradient
    gradients = {}
    radial_gradients = root.findall(".//svg:radialGradient", ns)
    linear_gradients = root.findall(".//svg:linearGradient", ns)

    # extract first stop-color in gradient
    for gradient in radial_gradients + linear_gradients:
        gradient_id = gradient.get("id")
        if gradient_id is not None:
            first_stop = gradient.xpath(".//svg:stop[1]/@stop-color", namespaces=ns)
            if first_stop:
                gradients[gradient_id] = first_stop[0]
            else:
                # no stop, then access xlink:href
                color = get_gradient_color(gradient_id, gradients, root, ns)
                if color:
                    gradients[gradient_id] = color

    # get all paths
    paths = root.findall(".//svg:path", ns)

    # Replace the fill reference in path
    for i, path in enumerate(paths):
        fill = path.get("fill")
        if fill and fill.startswith("url(#"):
            gradient_id = fill[5:-1]  # get gradient id
            if gradient_id in gradients:
                # replace fill
                path.set("fill", gradients[gradient_id])
        elif fill is None:
            if fill_is_empty == "previous":
                # If the current path does not fill, try to get the color from the previous valid path
                previous_colors = get_previous_fill_color(paths, i - 1)
                if previous_colors:
                    # Convert valid colors to RGB and calculate the average value
                    rgb_colors = [hex_to_rgb(color) for color in previous_colors]
                    average_hex_color = average_color(rgb_colors)
                    if average_hex_color:
                        path.set("fill", average_hex_color)
            elif fill_is_empty == "skip":
                continue
            else:  # 'default': black
                path.set("fill", "#fffff")

    # delete all gradient
    for gradient in radial_gradients + linear_gradients:
        root.remove(gradient)

    # return etree.tostring(root, encoding=str)
    return etree.tostring(root, encoding="utf-8").decode()


"""Delete the <svg> tag and keep the other tags"""


def remove_svg_tag(svg_string):
    root = ET.fromstring(svg_string)

    if root.tag == "{http://www.w3.org/2000/svg}svg":
        result = ""

        # remove namespace. xmlns="http://www.w3.org/2000/svg"
        for elem in root.iter():
            elem.tag = elem.tag.split("}", 1)[1] if "}" in elem.tag else elem.tag

        # tostring
        for elem in root:
            svg_element = ET.tostring(elem, encoding="unicode", method="xml")
            # NOTE: replace " />" with "/>"
            svg_element = svg_element.replace(" />", "/>")
            result += svg_element
            if not result.endswith("\n"):
                result += "\n"

        return result
    else:
        return svg_string


def remove_defs_tags(text):
    """Remove all <defs/> tags from the input text."""
    return text.replace("<defs/>", "")


if __name__ == "__main__":
    # svg_str = '<?xml version="1.0" ?><svg xmlns="http://www.w3.org/2000/svg" xmlns:ev="http://www.w3.org/2001/xml-events" xmlns:xlink="http://www.w3.org/1999/xlink" baseProfile="full" height="100%" version="1.1" viewBox="-215.0 1430.0 10793.088235294117 3500.0" width="100%"><defs/><path d="m765,-125l-20,125l-960,0l20,-125l280,-25q55,-5,85,-17q30,-12,60,-62q30,-50,85,-165l1505,-3005l235,0l455,3005q25,170,45,203q20,33,90,43l270,25l-20,125l-1090,0l20,-125l280,-25q85,-10,98,-42q13,-32,-12,-202l-100,-655l-1265,0l-325,655q-85,170,-85,203q0,33,80,43l270,25m135,-1075l1170,0l-270,-1800l-900,1800" transform="translate(0,4880)"/><path d="m620,-3275l25,-125l1095,0l-25,125l-275,25q-55,5,-80,18q-25,13,-40,60q-15,48,-35,168l-460,2610q-30,175,-20,205q10,30,85,40l270,25l-20,125l-1095,0l20,-125l280,-25q55,-5,80,-17q25,-12,40,-62q15,-50,35,-165l460,-2610q30,-175,20,-207q-10,-32,-85,-37l-275,-25" transform="translate(3110,4880)"/><path d="m620,-3275l25,-125l1260,0q425,0,690,205q265,205,365,553q100,348,20,783l-65,365q-80,440,-275,778q-195,338,-492,528q-297,190,-692,190l-1410,0l20,-125l280,-25q85,-10,105,-42q20,-32,50,-202l460,-2610q30,-175,20,-207q-10,-32,-85,-37l-275,-25m1285,25l-575,0l-550,3100l670,0q415,0,688,-277q273,-277,368,-817l155,-865q95,-535,-95,-837q-190,-302,-660,-302" transform="translate(4725,4880)"/><path d="m1175,50q-235,0,-457,-77q-222,-77,-375,-277q-152,-200,-172,-560l290,-90q-5,335,93,523q98,188,268,260q170,73,370,73q430,0,653,-202q223,-202,223,-537q0,-250,-135,-407q-135,-157,-450,-277l-315,-120q-315,-120,-482,-327q-167,-207,-167,-522q0,-250,120,-467q120,-217,365,-352q245,-135,620,-135q395,0,628,150q233,150,308,388q75,238,-20,498l-330,80q135,-470,-10,-720q-145,-250,-580,-250q-355,0,-570,183q-215,183,-215,538q0,230,123,383q123,153,438,273l315,120q330,125,503,323q173,198,173,508q0,250,-125,488q-125,238,-392,390q-267,153,-687,153" transform="translate(7990,4880)"/></svg>\n'
    svg_str = '<svg xmlns="http://www.w3.org/2000/svg" xmlns:ev="http://www.w3.org/2001/xml-events" xmlns:xlink="http://www.w3.org/1999/xlink" baseProfile="full" height="100%" version="1.1" viewBox="0 0 10 10" width="100%"><defs/><path d="m0 0l0 10l 10 0" transform="translate(0,0)"/></svg>'

    result_svg = remove_defs_tags(svg_str)
    result_svg = replace_gradient_tags(result_svg)
    result_svg = apply_g_attributes_to_children(result_svg)
    result_svg = remove_svg_tag(result_svg)
    print(result_svg)
