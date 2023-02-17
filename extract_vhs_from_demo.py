#!/usr/bin/python3

import vh_to_html
import lxml.etree
import lxml.html

import pickle as pkl
import os
import os.path

from typing import List, Dict
from typing import Any

demo_directory = "llmdemo-data/"
demo_list: List[str] = list( filter( lambda f: f.endswith(".pkl.prs")
                                   , os.listdir(demo_directory)
                                   )
                           )
vh_directory = "llmdemo-vh"

for dem in demo_list:
    with open(os.path.join(demo_directory, dem), "rb") as f:
        record_dict: Dict[str, Dict[str, Any]] = pkl.load(f)

    output_directory: str = os.path.join(vh_directory, dem[:-8])
    os.makedirs(output_directory, exist_ok=True)

    for i, st in enumerate(record_dict["trajectories"][0]):
        output_prefix = "step_{:d}.".format(i)

        #  View Hierarchy {{{ # 
        if "view_hierarchy" in st and st["view_hierarchy"] is not None:
            view_hierarchy: lxml.etree.Element = lxml.etree.fromstring(st["view_hierarchy"])

            with open( os.path.join( output_directory
                                   , output_prefix
                                   + "view_hierarchy"
                                   )
                     , "w"
                     ) as out_f:
                out_f.write( lxml.etree.tostring( view_hierarchy
                                                , pretty_print=True
                                                , encoding="unicode"
                                                )
                           )

            html_list: List[lxml.html.Element]
            bbox_list: List[List[int]]
            html_list, bbox_list = vh_to_html.convert_tree(view_hierarchy)

            with open( os.path.join( output_directory
                                   , output_prefix
                                   + "html"
                                   )
                     , "w"
                     ) as out_f:
                for html in html_list:
                    out_f.write( lxml.html.tostring( html
                                                   , pretty_print=True
                                                   , encoding="unicode"
                                                   )
                               )
                out_f.write("\n")
                for bb in bbox_list:
                    out_f.write("{:}\n".format(str(bb)))
        #  }}} View Hierarchy # 

        if "instruction" in st\
                and st["instruction"] is not None\
                and len(st["instruction"])>0:
            with open( os.path.join( output_directory
                                   , output_prefix
                                   + "instruction"
                                   )
                     , "w"
                     ) as out_f:
                out_f.write("\n".join(st["instruction"]) + "\n")
