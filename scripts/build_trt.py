# Prepare model for TensorRT inference.

import os
import numpy as np
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import argparse
import sys
import logging
from PIL import Image

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
log = logging.getLogger()
log.setLevel(logging.INFO)

class BatchLoader:

  def __init__(self, images, batch, imgsz):
    self.images = images
    self.batch = batch
    self.imgsz = imgsz
    self.current_batch = 0
    self.max_batches = len(self.images) // self.batch
        
  def read_and_preprocess(self, image_path):
     """Read and preprocess image to network input format"""
     with Image.open(image_path) as img:
      # Convert to RGB
      img = img.convert('RGB')
      # Resize to width * weight
      img = img.resize((self.imgsz, self.imgsz))
      img_array = np.array(img, dtype=np.float32)
      # Normalize to [0,1]
      img_array = img_array / 255.0
      # Change to NCHW format
      img_array = np.transpose(img_array, (2, 0, 1))
      return img_array

  def next_batch(self):
    """Get next batch of images"""
    if self.current_batch < self.max_batches:
      batch_files = self.images[self.current_batch * self.batch: (self.current_batch + 1) * self.batch]
      batch_data = np.zeros((self.batch, 3, self.imgsz, self.imgsz), dtype=np.float32)      
      for i, f in enumerate(batch_files):
        batch_data[i] = self.read_and_preprocess(f)         
      self.current_batch += 1
      return batch_data
    else:
      return None

class EntropyCalibrator(trt.IInt8EntropyCalibrator2):

  def __init__(self, calib_cache):
    trt.IInt8EntropyCalibrator2.__init__(self)
    self.cache_file = calib_cache
    self.batch_loader = None
    self.d_input = None

  def set_batch_loader(self, calib_data, calib_batch, imgsz):
    """Set batch loader. If cache is available this code is not executed"""
    images= [os.path.join(calib_data, f) for f in os.listdir(calib_data) 
             if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    self.batch_loader = BatchLoader(images, calib_batch, imgsz)
    if len(images) < self.batch_loader.batch:
      log.error(f"Too few images ({len(images)}) for the specified batch size ({self.batch_loader.batch})")
      sys.exit(1)
    self.d_input = cuda.mem_alloc(self.batch_loader.batch * 3 * imgsz * imgsz * np.dtype(np.float32).itemsize)

  def get_batch_size(self):
    if self.batch_loader:
      return self.batch_loader.batch
    return 1

  def get_batch(self, names):
    """Get a batch of input for calibration. If cache is available this code is not executed"""
    if not self.batch_loader:
      return None
    batch = self.batch_loader.next_batch()
    if batch is None:
      return None
    cuda.memcpy_htod(self.d_input, batch) # copy data from host (CPU) to device (GPU)
    return [int(self.d_input)]

  def read_calibration_cache(self):
    #log.info(f"Searching for calibration cache: {self.cache_file}")
    if os.path.exists(self.cache_file):
      #log.info(f"Reading calibration cache: {self.cache_file}")
      with open(self.cache_file, "rb") as f:
        return f.read()
    return None

  def write_calibration_cache(self, cache):
    #log.info(f"Writing calibration cache: {self.cache_file}")
    with open(self.cache_file, "wb") as f:
      f.write(cache)

def build(onnx_file, engine_file, precision, calib_data, calib_cache, calib_batch, imgsz, workspace):

  # Create TensorRT builder, config, network, and ONNX parser
  logger = trt.Logger(trt.Logger.INFO)
  builder = trt.Builder(logger)
  config = builder.create_builder_config()
  config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
  config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace << 30) # set max workspace (GB)
  # Modern TensorRT recommend using EXPLICIT_BATCH, which means the batch size is retrieved from the ONNX input
  # If the ONNX model was generated using ultralytics, batch size = 1 by defaul
  network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
  parser = trt.OnnxParser(network, logger) 

  # Parse ONNX file
  with open(onnx_file, 'rb') as f:
    if not parser.parse(f.read()):
      log.error(f"Failed to load ONNX file: {onnx_file}")
      for error in range(parser.num_errors):
        log.error(parser.get_error(error))
      sys.exit(1)
  # Show information
  inputs = [network.get_input(i) for i in range(network.num_inputs)]
  outputs = [network.get_output(i) for i in range(network.num_outputs)]   
  for input in inputs:
    log.info(f"Model {input.name} shape: {input.shape} {input.dtype}")
  for output in outputs:
    log.info(f"Model {output.name} shape: {output.shape} {output.dtype}") 
  
  # Set precision
  fp16 = builder.platform_has_fast_fp16
  int8 = builder.platform_has_fast_int8
  # FP16
  if precision == "fp16":
    if not fp16:
      log.warning(f"FP16 is not supported natively on this device")  
    else:
      config.set_flag(trt.BuilderFlag.FP16)
  # INT8
  elif precision == "int8":
    if not int8:
      log.warning(f"INT8 is not supported natively on this device. Trying to use FP16...")
      # Try FP16
      if not fp16:
        log.warning(f"FP16 is not supported natively on this device") 
    else:
      config.set_flag(trt.BuilderFlag.INT8)
      # Also enable FP16, as some layers may be even more efficient in FP16 than INT8
      if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        
    # Set INT8 calibrator
    config.int8_calibrator = EntropyCalibrator(calib_cache)
    if not os.path.exists(calib_cache):
      # At this point, if cache is unavailable images must exist
      config.int8_calibrator.set_batch_loader(calib_data, calib_batch, imgsz)
    
  # Build and save engine, performing calibration if needed
  serialized_engine = builder.build_serialized_network(network, config)
  if serialized_engine is None:
    log.error("Failed to build TensorRT engine")
    sys.exit(1)
  
  with open(engine_file, "wb") as f:
    f.write(serialized_engine)
  log.info(f"Engine saved to: {engine_file}")

if __name__ == "__main__":
    
  parser = argparse.ArgumentParser(description="Build and save a serialized engine file with FP16/INT8 precision")
  parser.add_argument("-o", "--onnx", type=str, help="Path to the input ONNX model")
  parser.add_argument("-e", "--engine", type=str, help="Path to the output engine model")
  parser.add_argument("-p", "--precision", default="fp16", choices=["fp32", "fp16", "int8"], help="The precision mode to build in, either 'fp32', 'fp16' or 'int8', default: 'fp16'")
  parser.add_argument("--calib-data", type=str, default="/home/jetson/inference/calib/images", help="Path to calibration data, default: /home/jetson/inference/calib/images")
  parser.add_argument("--calib-cache", type=str, default="/home/jetson/inference/calib/calib.cache", help="Path to calibration cache, default: /home/jetson/inference/calib/calib.cache")
  parser.add_argument("--calib-batch", type=int, default=16, help="Batch size for calibration, default: 16")
  parser.add_argument("--imgsz", type=int, default=640, help="Image size, default: 640")
  parser.add_argument("-w", "--workspace", type=int, default=4, help="Maximum workspace size, default: 4GB")
  args = parser.parse_args()
  
  if not all([args.onnx, args.engine]):
    parser.print_help()
    log.error("The follwing arguments are required: --onnx and --engine")
    sys.exit(1)
  
  log.info(f"Attempting to build '{args.engine}' from '{args.onnx}' with {args.precision} precision.")
  
  if args.precision == "int8":
    log.info(f"Calibration options: data={args.calib_data}, cache={args.calib_cache}, batch={args.calib_batch}, imgsz={args.imgsz}, workspace={args.workspace}GB") 
    if not (os.path.exists(args.calib_data) or os.path.exists(args.calib_cache)):
      parser.print_help()
      log.error(f"When INT8 is selected, user must provide valid calibration data (either images or cache)")
      sys.exit(1)
  
  build(
    args.onnx,
    args.engine,
    args.precision,
    args.calib_data,
    args.calib_cache,
    args.calib_batch,
    args.imgsz,
    args.workspace
  )