import tensorflow as tf
from tensorflow.python.tools import freeze_graph as fg
from model.utils import model_serializer
from google.protobuf import text_format
from tensorflow.python.framework import dtypes
from model.filter.iir import IIRFilter



class MotionDetector():
    def __init__(self, a, b, k, threshold, graph = None, nodeNamePrefix = ''):
        self.nodeNamePrefix = nodeNamePrefix +'mf/'
        if graph is None :
            self.graph = tf.Graph()
        else :
            self.graph = graph
        with self.graph.as_default():
            self.threshold = tf.constant(threshold, dtype=tf.float32, name=self.nodeNamePrefix + 'threshold')
            self.iirFilter = IIRFilter(a, b, self.graph, nodeNamePrefix = self.nodeNamePrefix)
            self.K = tf.constant(k, dtype=tf.float32, name=self.nodeNamePrefix + 'K')
            self.ax = tf.placeholder(dtype=tf.float32, name=self.nodeNamePrefix + 'ax')
            self.ay = tf.placeholder(dtype=tf.float32, name=self.nodeNamePrefix + 'ay')
            self.az = tf.placeholder(dtype=tf.float32, name=self.nodeNamePrefix + 'az')
            self.inState = self.iirFilter.inState
            #self.filteredA = tf.placeholder(dtype=tf.float32,name=self.nodeNamePrefix + 'filteredA')
        self.sess = tf.Session(graph = self.graph)
        self._y = None
        self._isMoving = None
    
    def export_for_mobile(self,export_path):
        tf.train.write_graph(self.graph.as_graph_def(), export_path, 'saved_model.pbtxt', as_text=True)
        #Optimize the graph
        model_serializer.optimize_for_inference(input = export_path + 'saved_model.pbtxt',
                          output = export_path + 'motion_detector.pb',
                          input_names =self.getInputNodeNames(),
                          placeholder_type_enum = dtypes.float32.as_datatype_enum,
                          output_names = self.getOutputNodeNames(),
                          frozen_graph = False)
   
    def getInputNodeNames(self):
        return self.nodeNamePrefix + 'ax' + \
                "," + self.nodeNamePrefix + "ay" + \
                "," + self.nodeNamePrefix + "az" + \
                "," + self.iirFilter.nodeNamePrefix + "InState" 
        
    def getOutputNodeNames(self):
        #'cf/cosDH,cf/sinDH,xiirFilter/iir/OutState,yiirFilter/iir/OutState'
        return self.nodeNamePrefix + 'a' + \
                "," + self.nodeNamePrefix + "y" + \
                "," + self.iirFilter.nodeNamePrefix + 'OutState'
    
    
    def export_for_visualization(self,export_path):
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter(export_path, self.graph)
        
    @property
    def y(self):
        if self._y is None:
            with self.graph.as_default():
                with tf.name_scope(self.nodeNamePrefix + "operations"):
                    a = tf.pow(self.ax,2) + tf.pow(self.ay,2) + tf.pow(self.az,2)
                    self.iirFilter.x = a
                aUnity = tf.multiply(a, 1.0, name = self.nodeNamePrefix + "a")
                filteredA = self.iirFilter.y
                with tf.name_scope(self.nodeNamePrefix + "operations"):
                    sub = tf.subtract(filteredA, self.threshold)
                    mulWithK = tf.multiply(self.K, sub)
                r = tf.sigmoid(mulWithK, name = self.nodeNamePrefix + "y")
                    #a = tf.tensordot([self.ax, self.ay, self.az],[self.ax, self.ay, self.az],1, name="mf/A")
            self._y = (r, self.iirFilter.outState)
        return self._y
    
    def run(self, ax, ay, az , inState):
        fetches = [self.y]
        a = self.sess.run(fetches, feed_dict = {self.ax : ax, self.ay : ay, self.az : az, self.iirFilter.inState : inState })
        return a