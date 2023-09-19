#!/usr/bin/python3
# Copyright 2023 SJTU X-Lance Lab
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Created by Danyang Zhang @X-Lance.

import lxml.etree
import lxml.html

from typing import Dict, List, Tuple, Pattern

from android_env.wrappers.vh_io_wrapper import filter_elements

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

    node_list: List[lxml.etree.Element]
    bbox_list: List[List[int]] = []
    node_list, bbox_list = filter_elements(node)

    result_list: List[lxml.html.Element] = []

    for i, n in enumerate(node_list):
        html_element: lxml.html.Element = convert_node(n)
        html_element.set("id", str(i))
        html_element.set("clickable", n.get("clickable"))
        result_list.append(html_element)

    return result_list, bbox_list
    #  }}} function convert_tree # 

def convert_simple_page(page: str) -> List[str]:
    """
    Args:
        page (str): " [SEP] " concatenated page observation

    Returns:
        List[str]: page observation devided at " [SEP] "
    """

    return page.split(" [SEP] ")

def simplify_html(page: str, with_eid: bool = False) -> List[str]:
    #  function simplify_html {{{ # 
    """
    Args:
        page (str): full html page observation
        with_eid (bool): if an auxiliary `eid` (element id) should be added to
          the returned elements

    Returns:
        List[str]: only leaf nodes of the html
    """

    page = page.replace("<br>", "&#10;")\
               .replace("<br/>", "&#10;")
    html_root: lxml.html.Element = lxml.html.fromstring(page)
    for n in list(html_root):
        if n.tag=="body":
            body_root: lxml.html.Element = n
            break
    result_list: List[str] = []

    if with_eid:
        id_counter = 0
    for n in body_root.iter():
        if isinstance(n, lxml.html.HtmlComment):
            continue
        if len(list(n))==0:
            if with_eid:
                n.set("eid", str(id_counter))
                id_counter += 1
            if "href" in n.attrib:
                del n.attrib["href"]
            if "data-url" in n.attrib:
                del n.attrib["data-url"]
            if "src" in n.attrib:
                del n.attrib["src"]
            result_list.append( lxml.html.tostring( n
                                                  , pretty_print=True
                                                  , encoding="unicode"
                                                  ).strip()\
                                                   .replace("\n", "&#10;")\
                                                   .replace("\r", "&#13;")
                              )
    return result_list
    #  }}} function simplify_html # 

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
