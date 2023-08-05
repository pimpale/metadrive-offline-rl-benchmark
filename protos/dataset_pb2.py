# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: protos/dataset.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from protos import label_pb2 as protos_dot_label__pb2
from protos import map_pb2 as protos_dot_map__pb2
from protos import vector_pb2 as protos_dot_vector__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x14protos/dataset.proto\x12\x12waymo.open_dataset\x1a\x12protos/label.proto\x1a\x10protos/map.proto\x1a\x13protos/vector.proto\"\x1b\n\x0bMatrixShape\x12\x0c\n\x04\x64ims\x18\x01 \x03(\x05\"O\n\x0bMatrixFloat\x12\x10\n\x04\x64\x61ta\x18\x01 \x03(\x02\x42\x02\x10\x01\x12.\n\x05shape\x18\x02 \x01(\x0b\x32\x1f.waymo.open_dataset.MatrixShape\"O\n\x0bMatrixInt32\x12\x10\n\x04\x64\x61ta\x18\x01 \x03(\x05\x42\x02\x10\x01\x12.\n\x05shape\x18\x02 \x01(\x0b\x32\x1f.waymo.open_dataset.MatrixShape\"l\n\nCameraName\"^\n\x04Name\x12\x0b\n\x07UNKNOWN\x10\x00\x12\t\n\x05\x46RONT\x10\x01\x12\x0e\n\nFRONT_LEFT\x10\x02\x12\x0f\n\x0b\x46RONT_RIGHT\x10\x03\x12\r\n\tSIDE_LEFT\x10\x04\x12\x0e\n\nSIDE_RIGHT\x10\x05\"]\n\tLaserName\"P\n\x04Name\x12\x0b\n\x07UNKNOWN\x10\x00\x12\x07\n\x03TOP\x10\x01\x12\t\n\x05\x46RONT\x10\x02\x12\r\n\tSIDE_LEFT\x10\x03\x12\x0e\n\nSIDE_RIGHT\x10\x04\x12\x08\n\x04REAR\x10\x05\"\x1e\n\tTransform\x12\x11\n\ttransform\x18\x01 \x03(\x01\"X\n\x08Velocity\x12\x0b\n\x03v_x\x18\x01 \x01(\x02\x12\x0b\n\x03v_y\x18\x02 \x01(\x02\x12\x0b\n\x03v_z\x18\x03 \x01(\x02\x12\x0b\n\x03w_x\x18\x04 \x01(\x01\x12\x0b\n\x03w_y\x18\x05 \x01(\x01\x12\x0b\n\x03w_z\x18\x06 \x01(\x01\"\xa3\x03\n\x11\x43\x61meraCalibration\x12\x31\n\x04name\x18\x01 \x01(\x0e\x32#.waymo.open_dataset.CameraName.Name\x12\x11\n\tintrinsic\x18\x02 \x03(\x01\x12\x30\n\textrinsic\x18\x03 \x01(\x0b\x32\x1d.waymo.open_dataset.Transform\x12\r\n\x05width\x18\x04 \x01(\x05\x12\x0e\n\x06height\x18\x05 \x01(\x05\x12g\n\x19rolling_shutter_direction\x18\x06 \x01(\x0e\x32\x44.waymo.open_dataset.CameraCalibration.RollingShutterReadOutDirection\"\x8d\x01\n\x1eRollingShutterReadOutDirection\x12\x0b\n\x07UNKNOWN\x10\x00\x12\x11\n\rTOP_TO_BOTTOM\x10\x01\x12\x11\n\rLEFT_TO_RIGHT\x10\x02\x12\x11\n\rBOTTOM_TO_TOP\x10\x03\x12\x11\n\rRIGHT_TO_LEFT\x10\x04\x12\x12\n\x0eGLOBAL_SHUTTER\x10\x05\"\xcd\x01\n\x10LaserCalibration\x12\x30\n\x04name\x18\x01 \x01(\x0e\x32\".waymo.open_dataset.LaserName.Name\x12\x19\n\x11\x62\x65\x61m_inclinations\x18\x02 \x03(\x01\x12\x1c\n\x14\x62\x65\x61m_inclination_min\x18\x03 \x01(\x01\x12\x1c\n\x14\x62\x65\x61m_inclination_max\x18\x04 \x01(\x01\x12\x30\n\textrinsic\x18\x05 \x01(\x0b\x32\x1d.waymo.open_dataset.Transform\"\xf6\x03\n\x07\x43ontext\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x42\n\x13\x63\x61mera_calibrations\x18\x02 \x03(\x0b\x32%.waymo.open_dataset.CameraCalibration\x12@\n\x12laser_calibrations\x18\x03 \x03(\x0b\x32$.waymo.open_dataset.LaserCalibration\x12\x30\n\x05stats\x18\x04 \x01(\x0b\x32!.waymo.open_dataset.Context.Stats\x1a\xa4\x02\n\x05Stats\x12J\n\x13laser_object_counts\x18\x01 \x03(\x0b\x32-.waymo.open_dataset.Context.Stats.ObjectCount\x12K\n\x14\x63\x61mera_object_counts\x18\x05 \x03(\x0b\x32-.waymo.open_dataset.Context.Stats.ObjectCount\x12\x13\n\x0btime_of_day\x18\x02 \x01(\t\x12\x10\n\x08location\x18\x03 \x01(\t\x12\x0f\n\x07weather\x18\x04 \x01(\t\x1aJ\n\x0bObjectCount\x12,\n\x04type\x18\x01 \x01(\x0e\x32\x1e.waymo.open_dataset.Label.Type\x12\r\n\x05\x63ount\x18\x02 \x01(\x05\"\xfd\x01\n\nRangeImage\x12\x1e\n\x16range_image_compressed\x18\x02 \x01(\x0c\x12$\n\x1c\x63\x61mera_projection_compressed\x18\x03 \x01(\x0c\x12#\n\x1brange_image_pose_compressed\x18\x04 \x01(\x0c\x12#\n\x1brange_image_flow_compressed\x18\x05 \x01(\x0c\x12%\n\x1dsegmentation_label_compressed\x18\x06 \x01(\x0c\x12\x38\n\x0brange_image\x18\x01 \x01(\x0b\x32\x1f.waymo.open_dataset.MatrixFloatB\x02\x18\x01\"\xe0\x02\n\x17\x43\x61meraSegmentationLabel\x12\x1e\n\x16panoptic_label_divisor\x18\x01 \x01(\x05\x12\x16\n\x0epanoptic_label\x18\x02 \x01(\x0c\x12q\n instance_id_to_global_id_mapping\x18\x03 \x03(\x0b\x32G.waymo.open_dataset.CameraSegmentationLabel.InstanceIDToGlobalIDMapping\x12\x13\n\x0bsequence_id\x18\x04 \x01(\t\x12\x1b\n\x13num_cameras_covered\x18\x05 \x01(\x0c\x1ah\n\x1bInstanceIDToGlobalIDMapping\x12\x19\n\x11local_instance_id\x18\x01 \x01(\x05\x12\x1a\n\x12global_instance_id\x18\x02 \x01(\x05\x12\x12\n\nis_tracked\x18\x03 \x01(\x08\"\xe4\x02\n\x0b\x43\x61meraImage\x12\x31\n\x04name\x18\x01 \x01(\x0e\x32#.waymo.open_dataset.CameraName.Name\x12\r\n\x05image\x18\x02 \x01(\x0c\x12+\n\x04pose\x18\x03 \x01(\x0b\x32\x1d.waymo.open_dataset.Transform\x12.\n\x08velocity\x18\x04 \x01(\x0b\x32\x1c.waymo.open_dataset.Velocity\x12\x16\n\x0epose_timestamp\x18\x05 \x01(\x01\x12\x0f\n\x07shutter\x18\x06 \x01(\x01\x12\x1b\n\x13\x63\x61mera_trigger_time\x18\x07 \x01(\x01\x12 \n\x18\x63\x61mera_readout_done_time\x18\x08 \x01(\x01\x12N\n\x19\x63\x61mera_segmentation_label\x18\n \x01(\x0b\x32+.waymo.open_dataset.CameraSegmentationLabel\"l\n\x0c\x43\x61meraLabels\x12\x31\n\x04name\x18\x01 \x01(\x0e\x32#.waymo.open_dataset.CameraName.Name\x12)\n\x06labels\x18\x02 \x03(\x0b\x32\x19.waymo.open_dataset.Label\"\xa1\x01\n\x05Laser\x12\x30\n\x04name\x18\x01 \x01(\x0e\x32\".waymo.open_dataset.LaserName.Name\x12\x32\n\nri_return1\x18\x02 \x01(\x0b\x32\x1e.waymo.open_dataset.RangeImage\x12\x32\n\nri_return2\x18\x03 \x01(\x0b\x32\x1e.waymo.open_dataset.RangeImage\"\xb8\x04\n\x05\x46rame\x12,\n\x07\x63ontext\x18\x01 \x01(\x0b\x32\x1b.waymo.open_dataset.Context\x12\x18\n\x10timestamp_micros\x18\x02 \x01(\x03\x12+\n\x04pose\x18\x03 \x01(\x0b\x32\x1d.waymo.open_dataset.Transform\x12/\n\x06images\x18\x04 \x03(\x0b\x32\x1f.waymo.open_dataset.CameraImage\x12)\n\x06lasers\x18\x05 \x03(\x0b\x32\x19.waymo.open_dataset.Laser\x12/\n\x0claser_labels\x18\x06 \x03(\x0b\x32\x19.waymo.open_dataset.Label\x12@\n\x16projected_lidar_labels\x18\t \x03(\x0b\x32 .waymo.open_dataset.CameraLabels\x12\x37\n\rcamera_labels\x18\x08 \x03(\x0b\x32 .waymo.open_dataset.CameraLabels\x12:\n\x0eno_label_zones\x18\x07 \x03(\x0b\x32\".waymo.open_dataset.Polygon2dProto\x12\x34\n\x0cmap_features\x18\n \x03(\x0b\x32\x1e.waymo.open_dataset.MapFeature\x12\x35\n\x0fmap_pose_offset\x18\x0b \x01(\x0b\x32\x1c.waymo.open_dataset.Vector3d*\t\x08\xe8\x07\x10\x80\x80\x80\x80\x02')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'protos.dataset_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _MATRIXFLOAT.fields_by_name['data']._options = None
  _MATRIXFLOAT.fields_by_name['data']._serialized_options = b'\020\001'
  _MATRIXINT32.fields_by_name['data']._options = None
  _MATRIXINT32.fields_by_name['data']._serialized_options = b'\020\001'
  _RANGEIMAGE.fields_by_name['range_image']._options = None
  _RANGEIMAGE.fields_by_name['range_image']._serialized_options = b'\030\001'
  _globals['_MATRIXSHAPE']._serialized_start=103
  _globals['_MATRIXSHAPE']._serialized_end=130
  _globals['_MATRIXFLOAT']._serialized_start=132
  _globals['_MATRIXFLOAT']._serialized_end=211
  _globals['_MATRIXINT32']._serialized_start=213
  _globals['_MATRIXINT32']._serialized_end=292
  _globals['_CAMERANAME']._serialized_start=294
  _globals['_CAMERANAME']._serialized_end=402
  _globals['_CAMERANAME_NAME']._serialized_start=308
  _globals['_CAMERANAME_NAME']._serialized_end=402
  _globals['_LASERNAME']._serialized_start=404
  _globals['_LASERNAME']._serialized_end=497
  _globals['_LASERNAME_NAME']._serialized_start=417
  _globals['_LASERNAME_NAME']._serialized_end=497
  _globals['_TRANSFORM']._serialized_start=499
  _globals['_TRANSFORM']._serialized_end=529
  _globals['_VELOCITY']._serialized_start=531
  _globals['_VELOCITY']._serialized_end=619
  _globals['_CAMERACALIBRATION']._serialized_start=622
  _globals['_CAMERACALIBRATION']._serialized_end=1041
  _globals['_CAMERACALIBRATION_ROLLINGSHUTTERREADOUTDIRECTION']._serialized_start=900
  _globals['_CAMERACALIBRATION_ROLLINGSHUTTERREADOUTDIRECTION']._serialized_end=1041
  _globals['_LASERCALIBRATION']._serialized_start=1044
  _globals['_LASERCALIBRATION']._serialized_end=1249
  _globals['_CONTEXT']._serialized_start=1252
  _globals['_CONTEXT']._serialized_end=1754
  _globals['_CONTEXT_STATS']._serialized_start=1462
  _globals['_CONTEXT_STATS']._serialized_end=1754
  _globals['_CONTEXT_STATS_OBJECTCOUNT']._serialized_start=1680
  _globals['_CONTEXT_STATS_OBJECTCOUNT']._serialized_end=1754
  _globals['_RANGEIMAGE']._serialized_start=1757
  _globals['_RANGEIMAGE']._serialized_end=2010
  _globals['_CAMERASEGMENTATIONLABEL']._serialized_start=2013
  _globals['_CAMERASEGMENTATIONLABEL']._serialized_end=2365
  _globals['_CAMERASEGMENTATIONLABEL_INSTANCEIDTOGLOBALIDMAPPING']._serialized_start=2261
  _globals['_CAMERASEGMENTATIONLABEL_INSTANCEIDTOGLOBALIDMAPPING']._serialized_end=2365
  _globals['_CAMERAIMAGE']._serialized_start=2368
  _globals['_CAMERAIMAGE']._serialized_end=2724
  _globals['_CAMERALABELS']._serialized_start=2726
  _globals['_CAMERALABELS']._serialized_end=2834
  _globals['_LASER']._serialized_start=2837
  _globals['_LASER']._serialized_end=2998
  _globals['_FRAME']._serialized_start=3001
  _globals['_FRAME']._serialized_end=3569
# @@protoc_insertion_point(module_scope)
