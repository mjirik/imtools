import imtools
import io3d
from dicom2fem.seg2fem import gen_mesh_from_voxels_mc, smooth_mesh
import time


fn = io3d.datasets.join_path("medical", "orig", "3Dircadb1.1", "MASKS_DICOM", "liver", get_root=True)
datap = io3d.read(fn)

data3d = datap["data3d"]
voxelsize_mm = datap["voxelsize_mm"]
seg = data3d > 0

seg = seg[::,::2,::2]
print(seg.shape)

t0 = time.time()
gen_mesh_from_voxels_mc(seg, voxelsize_mm)

t1 = time.time()
print("Time: ", t1-t0)


