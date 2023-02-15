#!/usr/bin/python3

import lxml.etree
import lxml.html

from typing import Dict, List, Tuple, Pattern

import re

def convert_node(node: lxml.etree.Element) -> lxml.html.Element:
    #  function convert_node {{{ # 
    """
    Converts one leaf node in android view hierarchy to html element. Will
    convert the class, text, resource-id, and content-desc properties.

    Args:
        node (lxml.etree.Element): leaf node from an android view hierarchy

    Returns:
        lxml.html.Element: the converted html element. usually is p, button,
          img, input, or div.
    """

    attribute_dict: Dict[str, str] = {}

    # convert resource-id
    resource_id: str = node.get("resource-id")
    if len(resource_id)>0:
        resource_identifyers = resource_id.rsplit("/", maxsplit=1)
        #assert len(resource_identifyers)==2
        attribute_dict["class"] = " ".join(resource_identifyers[-1].split("_"))

    # convert content-desc
    content_desc: str = node.get("content-desc")
    if len(content_desc)>0:
        attribute_dict["alt"] = content_desc

    # convert text
    text: str = node.get("text")

    # convert class
    vh_class_name: str = node.get("class")
    if vh_class_name.endswith("TextView"):
        html_element = lxml.html.Element( "p"
                                        , attribute_dict
                                        )
        if len(text)>0:
            html_element.text = text
    elif vh_class_name.endswith("Button")\
            or vh_class_name.endswith("MenuItemView"):
        html_element = lxml.html.Element( "button"
                                        , attribute_dict
                                        )
        if len(text)>0:
            html_element.text = text
    elif vh_class_name.endswith("ImageView")\
            or vh_class_name.endswith("IconView")\
            or vh_class_name.endswith("Image"):
        if len(text)>0:
            if "alt" in attribute_dict:
                attribute_dict["alt"] += ": " + text
            else:
                attribute_dict["alt"] = text
        html_element = lxml.html.Element( "img"
                                        , attribute_dict
                                        )
    elif vh_class_name.endswith("EditText"):
        if len(text)>0:
            attribute_dict["value"] = text
        attribute_dict["type"] = "text"
        html_element = lxml.html.Element( "input"
                                        , attribute_dict
                                        )
    else:
        html_element = lxml.html.Element( "div"
                                        , attribute_dict
                                        )
        if len(text)>0:
            html_element.text = text

    return html_element
    #  }}} function convert_node # 

bounds_pattern: Pattern[str] = re.compile(r"\[(\d+),(\d+)\]\[(\d+),(\d+)\]")
def convert_tree(node: lxml.etree.Element) ->\
        Tuple[ List[lxml.html.Element]
             , List[List[int]]
             ]:
    #  function convert_tree {{{ # 
    """
    Converts a view hierarchy tree to a html element list of all the leaf
    nodes.

    Args:
        node (lxml.etrn.Element): root node of the given view hierarchy tree

    Returns:
        List[lxml.html.Element]: the converted html element representation of
          the leaf nodes sorted in the docuemnt order
        List[List[int]]: list of list with length 4 of int as the corresponding
          bounding box of the leaf elements
    """

    result_list: List[lxml.html.Element] = []
    bbox_list: List[List[int]] = []

    id_counter = 0
    for n in node.iter():
        if n.getparent() is not None:
            n.set( "clickable"
                 , str(  n.get("clickable")=="true"\
                      or n.getparent().get("clickable")=="true"
                      ).lower()
                 )
        if n.get("bounds")=="[0,0][0,0]":
            continue
        if len(list(n))==0:
            html_element: lxml.html.Element = convert_node(n)
            html_element.set("id", str(id_counter))
            html_element.set("clickable", n.get("clickable"))
            result_list.append(html_element)
            id_counter += 1

            bounds_match = bounds_pattern.match(n.get("bounds"))
            bbox_list.append( list( map( int
                                       , bounds_match.groups()
                                       )
                                  )
                            )
    return result_list, bbox_list
    #  }}} function convert_tree # 

if __name__ == "__main__":
    import sys

    input_file: str = sys.argv[1]
    output_file: str = sys.argv[2]

    html_elements: List[lxml.html.Element]
    node_bboxes: List[List[int]]

    vh_tree: lxml.etree.ElementTree = lxml.etree.parse(input_file)
    html_elements, node_bboxes = convert_tree(vh_tree.getroot())

    with open(output_file, "w") as f:
        for html_elm, n_bb in zip(html_elements, node_bboxes):
            f.write( lxml.html.tostring( html_elm
                                       , pretty_print=True
                                       , encoding="unicode"
                                       ).strip()\
                   + " "
                   + str(n_bb)
                   + "\n"
                   )
