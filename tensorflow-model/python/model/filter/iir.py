import tensorflow as tf
from tensorflow.python.tools import freeze_graph as fg
from model.utils import model_serializer
from google.protobuf import text_format
from tensorflow.python.framework import dtypes

class IIRFilter:
    def __init__(self, a, b, graph = None, nodeNamePrefix = ''):
        self.nodeNamePrefix = nodeNamePrefix + 'iir/'
        if graph is None :
            self.graph = tf.Graph()
        else :
            self.graph = graph
        with self.graph.as_default():
            self.A = tf.constant(a, dtype=tf.float32, name=self.nodeNamePrefix + 'constA')
            self.B = tf.constant(b, dtype=tf.float32, name=self.nodeNamePrefix + 'constB')
            self.order = len(b) - 1
            self.x = tf.placeholder(dtype=tf.float32, name=self.nodeNamePrefix + 'X')
            self.inState = tf.placeholder(dtype=tf.float32, name=self.nodeNamePrefix + 'InState') # I think we don't need that
            self.dummy = tf.Variable(tf.zeros([2]), name="dummy")
            self.init = tf.global_variables_initializer()
        self.sess = tf.Session(graph = self.graph)
        self._y = None
        self._outState = None
    
    def export_for_mobile(self,export_path):
        tf.train.write_graph(self.graph.as_graph_def(), export_path, 'saved_model.pbtxt', as_text=True)
        '''
        saver.save(sess, export_path + "checkpoint/saved_model.ckpt") # writing dummy variable checkpoint
    
        fg.freeze_graph(input_graph = export_path + 'saved_model.pbtxt', 
             input_binary = False, 
             output_graph = export_path + 'frozen_saved_model.pb',
             input_checkpoint = export_path + "checkpoint/saved_model.ckpt",
             input_saver = None,
             restore_op_name = None,
             filename_tensor_name = None,
             clear_devices = None,
             initializer_nodes = None,
             output_node_names = 'iir/OutState,iir/Y')
        '''
        
        #Optimize the graph
        model_serializer.optimize_for_inference(input = export_path + 'saved_model.pbtxt',
                          output = export_path + 'iir.pb',
                          input_names = self.nodeNamePrefix + 'X',
                          placeholder_type_enum = dtypes.float32.as_datatype_enum,
                          output_names = self.nodeNamePrefix + 'OutState' +','+ self.nodeNamePrefix + 'Y',
                          frozen_graph = False)
        
        
    def export_for_visualization(self,export_path):
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter(export_path, self.graph)
    
    @property
    def y(self):
        if self._y is None:
            with self.graph.as_default():
                with tf.name_scope(name=self.nodeNamePrefix +"iirOperations"):
                    bMulX = tf.multiply(self.B[0], self.x)
                    firstElemInState = self.inState[0]
                y = tf.add(bMulX , firstElemInState, name = self.nodeNamePrefix + "Y")
                self._y = y
        return self._y
    
    @property
    def outState(self):
        if self._outState is None:
            with self.graph.as_default():
                with tf.name_scope(name=self.nodeNamePrefix +"iirOperations"):
                    oState = tf.multiply(self.B[1:], self.x) -  tf.multiply(self.A[1:], self.y)
                    slicedOutState = oState[:-1]
                    slicedOutState = slicedOutState + self.inState[1:]
                    lastOutState = oState[self.order-1 :]
                oState = tf.concat([slicedOutState, lastOutState],0, name = self.nodeNamePrefix + "OutState")
                self._outState = oState
        return self._outState
    
    def run(self, xIn, inState):
        fetches = [self.y, self.outState]
        outs = self.sess.run(fetches, feed_dict = {self.x : xIn, self.inState : inState})
        return outs
