
from pathlib import Path
import os, sys, fire, time
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from dufomap import dufomap
from dufomap.utils import pcdpy3
from dufomap.utils.o3d_view import MyVisualizer
BASE_DIR = os.path.abspath(os.path.join( os.path.dirname( __file__ ), '' ))
sys.path.append(BASE_DIR)
VIEW_FILE = f"{BASE_DIR}/assets/view/demo.json"

def xyzqwxyz_to_matrix(xyzqwxyz: list):
    """
    input: xyzqwxyz: [x, y, z, qx, qy, qz, qw] a list of 7 elements
    """
    rotation = R.from_quat([xyzqwxyz[4], xyzqwxyz[5], xyzqwxyz[6], xyzqwxyz[3]]).as_matrix()
    pose = np.eye(4).astype(np.float64)
    pose[:3, :3] = rotation
    pose[:3, 3] = xyzqwxyz[:3]
    return pose
def inv_pose_matrix(pose):
    inv_pose = np.eye(4)
    inv_pose[:3, :3] = pose[:3, :3].T
    inv_pose[:3, 3] = -pose[:3, :3].T.dot(pose[:3, 3])
    return inv_pose
class DynamicMapData:
    def __init__(self, directory):
        self.scene_id = directory.split("/")[-1]
        self.directory = Path(directory) / "pcd"
        self.pcd_files = [os.path.join(self.directory, f) for f in sorted(os.listdir(self.directory)) if f.endswith('.pcd')]

    def __len__(self):
        return len(self.pcd_files)
    
    def __getitem__(self, index_):
        res_dict = {
            'scene_id': self.scene_id,
            'timestamp': self.pcd_files[index_].split("/")[-1].split(".")[0],
        }
        pcd_ = pcdpy3.PointCloud.from_path(self.pcd_files[index_])
        pc0 = pcd_.np_data[:,:3]
        pose0 = xyzqwxyz_to_matrix(list(pcd_.viewpoint))
        inv_pose0 = inv_pose_matrix(pose0)
        ego_pc0 = pc0 @ inv_pose0[:3, :3].T + inv_pose0[:3, 3]
        res_dict['pc'] = ego_pc0.astype(np.float32)
        res_dict['pose'] = pose0
        return res_dict


def main_vis(
    data_dir: str = "/home/kin/data/00",
):
    dataset = DynamicMapData(data_dir)
    o3d_vis = MyVisualizer(view_file=VIEW_FILE, window_title=f"view dufomap, `SPACE` start/stop")
    opt = o3d_vis.vis.get_render_option()
    opt.point_size = 3

    # STEP 0: initialize 
    mydufo = dufomap(0.1, 0.1, 2)

    for data_id in (pbar := tqdm(range(0, len(dataset)),ncols=100)):
        data = dataset[data_id]
        now_scene_id = data['scene_id']
        pbar.set_description(f"id: {data_id}, scene_id: {now_scene_id}, timestamp: {data['timestamp']}")
        
        # STEP 1: integrate point cloud into dufomap
        mydufo.run(data['pc'], data['pose'], cloud_transform = True)

        # STEP 2: results
        label = mydufo.segment(data['pc'], data['pose'], cloud_transform = True).astype(np.uint8)


        # visualize, no need to view following code in detail
        static_pcd = o3d.geometry.PointCloud()
        static_pcd.points = o3d.utility.Vector3dVector(data['pc'][label == 0])
        dynamic_pcd = o3d.geometry.PointCloud()
        dynamic_pcd.points = o3d.utility.Vector3dVector(data['pc'][label == 1])
        dynamic_pcd.paint_uniform_color([0, 0, 0])
        o3d_vis.update([static_pcd, dynamic_pcd, o3d.geometry.TriangleMesh.create_coordinate_frame(size=2)])
    
    mydufo.printDetailTiming()

if __name__ == "__main__":
    start_time = time.time()
    fire.Fire(main_vis)
    print(f"Time used: {time.time() - start_time:.2f} s")