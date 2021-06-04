import tensorrt as trt
import pycuda.driver as cuda
import numpy as np
import pycuda.autoinit
import cv2
from PIL import Image
import inference as inf
import sys, time
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
trt_runtime = trt.Runtime(TRT_LOGGER)




def build_engine(onnx_path, shape = [1,250,250,3]):

   """
   This is the function to create the TensorRT engine
   Args:
      onnx_path : Path to onnx_file.
      shape : Shape of the input of the ONNX file.
  """
   EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
   with trt.Builder(TRT_LOGGER) as builder, builder.create_builder_config() as config,builder.create_network(EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
       config.max_workspace_size = (256 << 20)
       with open(onnx_path, 'rb') as model:
           parser.parse(model.read())
       network.get_input(0).shape = shape
       engine = builder.build_engine(network,config)
       return engine

def save_engine(engine, file_name):
   buf = engine.serialize()
   with open(file_name, 'wb') as f:
       f.write(buf)
def load_engine(trt_runtime, plan_path):
   with open(engine_path, 'rb') as f:
       engine_data = f.read()
   engine = trt_runtime.deserialize_cuda_engine(engine_data)
   return engine

def allocate_buffers(engine, batch_size, data_type):

   """
   This is the function to allocate buffers for input and output in the device
   Args:
      engine : The path to the TensorRT engine.
      batch_size : The batch size for execution time.
      data_type: The type of the data for input and output, for example trt.float32.

   Output:
      h_input_1: Input in the host.
      d_input_1: Input in the device.
      h_output_1: Output in the host.
      d_output_1: Output in the device.
      stream: CUDA stream.

   """

   # Determine dimensions and create page-locked memory buffers (which won't be swapped to disk) to hold host inputs/outputs.
   h_input_1 = cuda.pagelocked_empty(batch_size * trt.volume(engine.get_binding_shape(0)), dtype=trt.nptype(data_type))
   #print(engine.get_binding_shape(0))
   h_output = cuda.pagelocked_empty(batch_size * trt.volume(engine.get_binding_shape(1)), dtype=trt.nptype(data_type))
   #print(engine.get_binding_shape(1))
   # Allocate device memory for inputs and outputs.
   d_input_1 = cuda.mem_alloc(h_input_1.nbytes)

   d_output = cuda.mem_alloc(h_output.nbytes)
   # Create a stream in which to copy inputs/outputs and run inference.
   stream = cuda.Stream()
   return h_input_1, d_input_1, h_output, d_output, stream
def load_images_to_buffer(pics, pagelocked_buffer):
   preprocessed = np.asarray(pics).ravel()
   np.copyto(pagelocked_buffer, preprocessed)

def do_inference(engine, pics_1, h_input_1, d_input_1, h_output, d_output, stream, batch_size, height, width):
   """
   This is the function to run the inference
   Args:
      engine : Path to the TensorRT engine
      pics_1 : Input images to the model.
      h_input_1: Input in the host
      d_input_1: Input in the device
      h_output_1: Output in the host
      d_output_1: Output in the device
      stream: CUDA stream
      batch_size : Batch size for execution time
      height: Height of the output image
      width: Width of the output image

   Output:
      The list of output images

   """

   load_images_to_buffer(pics_1, h_input_1)

   with engine.create_execution_context() as context:
       # Transfer input data to the GPU.
       cuda.memcpy_htod_async(d_input_1, h_input_1, stream)

       # Run inference.

       context.profiler = trt.Profiler()
       context.execute(batch_size=1, bindings=[int(d_input_1), int(d_output)])

       # Transfer predictions back from the GPU.
       cuda.memcpy_dtoh_async(h_output, d_output, stream)
       # Synchronize the stream
       stream.synchronize()
       # Return the host output.
       #out = h_output
       return h_output

engine=build_engine('/home/fabricio/Documentos/embebidos2/Vitis-AI/Vitis-Tutorials/Prueba02/files/f_model_projectBP2.onnx', shape = [1,250,250,3])
#save_engine(engine, 'own_model.plan')
cap = cv2.VideoCapture('/home/fabricio/Documentos/embebidos2/Vitis-AI/Vitis-Tutorials/Prueba02/files/videos/hamburgers.mp4')
#cap = cv2.VideoCapture(0)
ret, frame = cap.read()
prev_img=frame
prev_img_res=cv2.resize(prev_img,(250,250))
prev_img_c=cv2.cvtColor(prev_img_res, cv2.COLOR_BGR2GRAY)
counter=0
acc=0.0
start_time =0
prev_frame_time = 0
while(True):

    ret, frame = cap.read()
    curr_img_res=cv2.resize(frame,(250,250))
    curr_img_res=curr_img_res/255.0
    #curr_img_c=cv2.cvtColor(curr_img_res, cv2.COLOR_BGR2GRAY)
    
    #flow = cv2.calcOpticalFlowFarneback(prev_img_c,curr_img_c, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    #vert = cv2.normalize(curr_img_res[...,1], None, 0, 255, cv2.NORM_MINMAX)
    #vert = vert.astype('uint8')
    #kernel = np.ones((15,15),np.float32)/225
    #smoothed = cv2.filter2D(vert,-1,kernel)

    #back = cv2.cvtColor(vert,cv2.COLOR_GRAY2RGB)

    h_input_1, d_input_1, h_output, d_output, stream = allocate_buffers(engine, 1, trt.float32)

    out = do_inference(engine, curr_img_res, h_input_1, d_input_1, h_output, d_output, stream, 1, 250, 250)
    #acc=acc+(time.time()-start_time)
    start_time =time.time()
    fps = 1/(start_time-prev_frame_time)
    prev_frame_time = start_time
    fps=int(fps)
    fps=str(fps)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, fps, (7, 70), font, 1, (100, 255, 0),3, cv2.LINE_AA)
    cv2.imshow("frame", curr_img_res)
    cv2.waitKey(1)
    #prev_img_c=curr_img_res
    acc=acc+(time.time()-start_time)
    print(" FPS: ", 1.0/(time.time()-start_time))
    print('******************')       
    print('\nINFERENCE:\n',out)
    print('\n******************')   
    print('\nFPS:',fps)
    print('\n******************')
    counter+=1
    #time.sleep(0.05)
    #if counter>=885:
    #    #print('******************')       
    #    print('\nFPS:',fps)
    #    print('\n******************')       
    #    break
#print(out)
