from tensorflow.python.tools import optimize_for_inference_lib
from tensorflow.core.framework import graph_pb2
from tensorflow.python.platform import gfile
from google.protobuf import text_format
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import graph_io
import os

def optimize_for_inference(input, output, input_names, placeholder_type_enum, output_names, frozen_graph, output_as_text = False ):
    input_graph_def = graph_pb2.GraphDef()
    with gfile.Open(input, "rb") as f:
        data = f.read()
    if frozen_graph:
        input_graph_def.ParseFromString(data)
    else:
        text_format.Merge(data.decode("utf-8"), input_graph_def)

    output_graph_def = optimize_for_inference_lib.optimize_for_inference(
      input_graph_def,
      input_names.split(","),
      output_names.split(","), 
      placeholder_type_enum
      )

    if output_as_text:
        graph_io.write_graph(output_graph_def,
                         os.path.dirname(output),
                         os.path.basename(output))
    else:
        f = gfile.FastGFile(output, "w")
        f.write(output_graph_def.SerializeToString())
    return 0