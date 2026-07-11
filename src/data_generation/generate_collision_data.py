import sys
sys.path.append('../')
import random
import numpy as np
import torch
import argparse
import os 
import h5py
import genesis as gs
import gc
import json
import trimesh

from utils.seeding import seed_everything
from utils.loading import load_mesh
from utils.visualization import save_pointcloud_video_genesis as save_pointcloud_video
from utils.sample import sample_points_on_mesh, sample_direction_hemisphere
from utils.transform import normalize_points_genesis as normalize_points
from torch_cluster import fps

from tqdm import tqdm

from multiprocessing import Process

def generate_initial_conditions(scenario_type, obj_size1=0.15, obj_size2=0.15):
    """
    Generates initial positions and velocities for a pair of objects.
    Space bounds: X, Y in [-0.5, 0.5], Z in [0.25, 0.5].
    """
    g = 9.81
    
    # Randomize positions
    p1 = np.random.uniform([-0.5, -0.5, 0.25], [0.5, 0.5, 0.5])
    p2 = np.random.uniform([-0.5, -0.5, 0.25], [0.5, 0.5, 0.5])
    
    # Ensure they don't start overlapping
    if np.linalg.norm(p1 - p2) < (obj_size1 + obj_size2):
        return None, None, None, None, None, None
        
    # Random angular velocities
    w1 = np.random.uniform(-2*np.pi, 2*np.pi, 3)
    w2 = np.random.uniform(-2*np.pi, 2*np.pi, 3)
    
    if scenario_type == 1:
        # Scenario 1: No collision
        v1 = np.random.uniform(-2.0, 2.0, 3)
        v2 = np.random.uniform(-2.0, 2.0, 3)
        
        # Check if they collide
        dp = p1 - p2
        dv = v1 - v2
        
        dv_norm2 = np.dot(dv, dv)
        if dv_norm2 > 1e-6:
            t_closest = -np.dot(dp, dv) / dv_norm2
            if t_closest > 0:
                min_dist = np.linalg.norm(dp + dv * t_closest)
                if min_dist < (obj_size1 + obj_size2) * 0.75:
                    # They might collide, resample
                    return None, None, None, None, None, None
        
        return p1, p2, v1, v2, w1, w2
        
    elif scenario_type == 2:
        # Scenario 2: One moving, one stationary
        v2 = np.zeros(3)
        
        # Object 2 hits floor at t_f
        t_f = np.sqrt(2 * p2[2] / g)
        
        # Choose collision time before obj 2 hits floor
        if t_f < 0.15:
            return None, None, None, None, None, None # obj 2 too low, resample positions
            
        # Target offset for glancing blow
        offset = np.random.uniform(-1, 1, 3)
        offset = (offset / np.linalg.norm(offset)) * np.random.uniform(0, ((obj_size1 + obj_size2) / 2) * 0.8)
        
        # To ensure velocity is <= 2.0, t_c must be large enough
        dp = p1 - p2
        min_tc = max(0.1, np.linalg.norm(offset - dp) / 2.0)
        if min_tc > t_f - 0.05:
            return None, None, None, None, None, None
            
        t_c = np.random.uniform(min_tc, t_f - 0.05)
        
        # v1 required to hit obj 2 at t_c with offset
        dv = (offset - dp) / t_c
        v1 = v2 + dv
        
        # Enforce 2 m/s speed limit
        if np.linalg.norm(v1) > 2.0:
            return None, None, None, None, None, None
        
        # Check if obj 1 hits floor before t_c
        z_1_tc = p1[2] + v1[2] * t_c - 0.5 * g * t_c**2
        if z_1_tc < 0:
            return None, None, None, None, None, None
            
        return p1, p2, v1, v2, w1, w2
        
    elif scenario_type == 3:
        # Scenario 3: Both moving and collide
        # Ensure both have time to fall
        t_f1 = np.sqrt(2 * p1[2] / g)
        t_f2 = np.sqrt(2 * p2[2] / g)
        max_tc = min(t_f1, t_f2) - 0.05
        
        if max_tc < 0.15:
            return None, None, None, None, None, None
            
        offset = np.random.uniform(-1, 1, 3)
        offset = (offset / np.linalg.norm(offset)) * np.random.uniform(0, ((obj_size1 + obj_size2) / 2) * 0.8)
        
        dp = p1 - p2
        
        # To ensure velocities <= 2.0, t_c must be large enough
        min_tc = max(0.1, np.linalg.norm(offset - dp) / 2.0)
        if min_tc > max_tc:
            return None, None, None, None, None, None
            
        t_c = np.random.uniform(min_tc, max_tc)
        
        dv = (offset - dp) / t_c
        
        v_base = np.random.uniform(-0.5, 0.5, 3)
        
        v1 = v_base + dv / 2
        v2 = v_base - dv / 2
        
        # Enforce 2 m/s speed limit
        if np.linalg.norm(v1) > 2.0 or np.linalg.norm(v2) > 2.0:
            return None, None, None, None, None, None
        
        # Check floor collision before t_c
        z_1_tc = p1[2] + v1[2] * t_c - 0.5 * g * t_c**2
        z_2_tc = p2[2] + v2[2] * t_c - 0.5 * g * t_c**2
        
        if z_1_tc < 0 or z_2_tc < 0:
            return None, None, None, None, None, None
            
        return p1, p2, v1, v2, w1, w2

    ### TODO: Scenario 4. Moving object hitting a still object on floor

    else:
        raise ValueError("Invalid scenario type")

# Method 1: Accurate barycentric coordinate calculation
def compute_barycentric_coordinates(points, triangles):
    """Compute exact barycentric coordinates for points on triangles."""
    points = np.asarray(points)
    triangles = np.asarray(triangles)
    
    if points.ndim == 1:
        points = points[np.newaxis, :]
        triangles = triangles[np.newaxis, :, :]
        
    v0 = triangles[:, 0, :]
    v1 = triangles[:, 1, :]
    v2 = triangles[:, 2, :]
    
    v0v1 = v1 - v0
    v0v2 = v2 - v0
    v0p = points - v0
    
    d00 = np.sum(v0v1 * v0v1, axis=1)
    d01 = np.sum(v0v1 * v0v2, axis=1)
    d11 = np.sum(v0v2 * v0v2, axis=1)
    d20 = np.sum(v0p * v0v1, axis=1)
    d21 = np.sum(v0p * v0v2, axis=1)
    
    denom = d00 * d11 - d01 * d01
    
    # Handle degenerate triangles
    valid = np.abs(denom) > 1e-10
    
    v = np.zeros_like(denom)
    w = np.zeros_like(denom)
    
    v[valid] = (d11[valid] * d20[valid] - d01[valid] * d21[valid]) / denom[valid]
    w[valid] = (d00[valid] * d21[valid] - d01[valid] * d20[valid]) / denom[valid]
    
    u = 1.0 - v - w
    
    # For invalid triangles, just use center
    u[~valid] = 1.0 / 3.0
    v[~valid] = 1.0 / 3.0
    w[~valid] = 1.0 / 3.0
    
    barycentric = np.stack([u, v, w], axis=1)
    
    if barycentric.shape[0] == 1 and points.ndim == 1:
        return barycentric[0]
    return barycentric

def sample_points_with_face_tracking(mesh, n_points):
    """Sample points on mesh and track their face and barycentric coordinates accurately"""
    # Sample points
    points, face_indices = trimesh.sample.sample_surface(mesh, n_points)
    
    # Get the triangles for each sampled point
    triangles = mesh.triangles[face_indices]
    
    # Calculate accurate barycentric coordinates for each point
    barycentric_coords = compute_barycentric_coordinates(points, triangles)
    
    return points, face_indices, barycentric_coords

def find_corresponding_points(transformed_mesh, face_indices, barycentric_coords):
    """Find corresponding points on the transformed mesh using face indices and barycentric coordinates"""
    transformed_triangles = transformed_mesh.triangles[face_indices]
    
    # Apply barycentric coordinates to transformed triangle
    b_coords = barycentric_coords[:, :, np.newaxis] # shape: (N, 3, 1)
    transformed_points = np.sum(b_coords * transformed_triangles, axis=1)
    
    return transformed_points

def run_generation(args):

    device = 'cpu' if args.cpu else 'cuda'
    gs.init(backend=gs.cpu if args.cpu else gs.cuda)

    # Temporally fixed parameters
    N = 2048
    TOTAL_SIZE = 5 # Size to normalize objects positions by. Objects are initialized within a 1 meter cube, but normalized by 5 meters to account for more space.
    OBJ_SIZE1 = 0.075 # Size of object 1 in meters
    OBJ_SIZE2 = 0.075 # Size of object 2 in meters
    RHO1 = 600
    FRICTION1 = 0.5
    RHO2 = 600
    FRICTION2 = 0.5
    FLOOR_FRICTION = 1.0
    
    # Load objects config & get obj list 
    if args.dataset_type == 'objaverse':
        data_dir = f'{args.base_dir}/{args.dataset_type}/raw/hf-objaverse-v1/glbs'
        with open(args.uid_list, "r") as f:
            obj_list = json.load(f)
        suffix = '.glb'
    elif args.dataset_type == '3d_primitives':
        data_dir = f'{args.base_dir}/{args.dataset_type}/glbs'
        with open(args.uid_list, "r") as f:
            obj_list = json.load(f)
        suffix = '.glb'
    else:
        raise NotImplementedError()
        
    start_idx = max(args.start_idx, 0)
    end_idx = min(args.end_idx, len(obj_list))
    idx_list = list(range(start_idx, end_idx))
    random.shuffle(idx_list)

    output_dir = f'{args.base_dir}/{args.dataset_type}/{args.output_dir}'
    os.makedirs(f'{output_dir}/h5', exist_ok=True)
    if args.visualization:
        os.makedirs(f'{output_dir}/visualization', exist_ok=True)

    for k, i in enumerate(idx_list):
        
        obj_path1 = obj_list[i]
        next_idx = idx_list[(k + 1) % len(idx_list)]
        obj_path2 = obj_list[next_idx]
        
        if not os.path.exists(f'{data_dir}/{obj_path1}{suffix}') or not os.path.exists(f'{data_dir}/{obj_path2}{suffix}'):
            continue
        
        jdx_list = list(range(args.n_samples))
        random.shuffle(jdx_list)
        
        for j in jdx_list:
            
            if device == 'cuda':
                torch.cuda.empty_cache()
            gc.collect() 
            
            output_idx = f'{i:05d}_{next_idx:05d}_{j:03d}_{args.scenario}'
            print(f'Generating {output_idx}...')
            output_path = f'{output_dir}/h5/{output_idx}.h5'
            print(output_path)
            if os.path.exists(output_path):
                continue
            
            seed_everything(output_idx)
            
            mesh1 = load_mesh(f'{data_dir}/{obj_path1}{suffix}') 
            mesh1.merge_vertices()
            mesh1.vertices, R1 = normalize_points(mesh1.vertices, size=OBJ_SIZE1, output_center=[0.0, 0.0, 0.0], random_rotation='uniform')
            mesh1.export(f'{data_dir}/{obj_path1}_normalized{suffix}')

            mesh2 = load_mesh(f'{data_dir}/{obj_path2}{suffix}') 
            mesh2.merge_vertices()
            mesh2.vertices, R2 = normalize_points(mesh2.vertices, size=OBJ_SIZE2, output_center=[0.0, 0.0, 0.0], random_rotation='uniform')
            mesh2.export(f'{data_dir}/{obj_path2}_normalized{suffix}')

            # min_height = np.min(mesh.vertices[:, 1])
            floor_height = 0.0
            gravity = 1 

            scene = gs.Scene(
                sim_options=gs.options.SimOptions(
                        dt=0.0417, 
                        substeps=20,
                        gravity=(0, 0, -9.81),  
                    ),
                vis_options=gs.options.VisOptions(
                    shadow=False,
                ),
                # Add this line to enable the IPC solver
                # coupler_options=gs.options.IPCCouplerOptions(
                #     enable_rigid_rigid_contact=True,
                # ),
                show_viewer=args.show_viewer,
                show_FPS=args.show_viewer
            )
                
            plane = scene.add_entity(
                morph=gs.morphs.Box(
                    pos=(0, 0, -0.05),      # Positioned slightly below Z=0
                    size=(10.0, 10.0, 0.1), # 10x10 meters, 10cm thick
                    fixed=True
                ),
                material=gs.materials.Rigid(
                    # needs_coup=True,        # This is now supported because it's a Box (Mesh)
                    # coup_restitution=0.5,   # Floor bounciness
                    # coup_friction=1.0,      # Floor friction
                    friction=FLOOR_FRICTION,
                ),
            )
            attempts = 0
            while True:
                attempts += 1
                print(f"Attempt {attempts}:")
                init_p1, init_p2, init_v1, init_v2, init_w1, init_w2 = generate_initial_conditions(args.scenario, obj_size1=OBJ_SIZE1, obj_size2=OBJ_SIZE2)
                
                if init_p1 is not None:
                    break
            print(f"Initial conditions generated for {output_idx}:")
            print(f"p1: {init_p1}, p2: {init_p2}, v1: {init_v1}, v2: {init_v2}, w1: {init_w1}, w2: {init_w2}")

            try:
                object1 = scene.add_entity(
                    morph=gs.morphs.Mesh(
                        file=f'{data_dir}/{obj_path1}_normalized{suffix}',
                        scale=1.0,
                        pos=init_p1.tolist(), 
                        euler=(0.0, 0.0, 0.0),
                        decompose_object_error_threshold=0.01,
                        recompute_inertia = True,
                    ),
                    surface=gs.surfaces.Rough(color=(0.1, 0.1, 0.9)),
                    material=gs.materials.Rigid(
                        # needs_coup=True,           # Required for IPC to see this object
                        # coup_restitution=0.5,      # This is your bounciness (0.0 to 1.0)
                        # coup_friction=0.5,         # Add this to control friction in IPC
                        friction=FRICTION1,              # Keep this for general rigid solver consistency
                        rho=RHO1,
                    ),
                )
                object2 = scene.add_entity(
                    morph=gs.morphs.Mesh(
                        file=f'{data_dir}/{obj_path2}_normalized{suffix}',
                        scale=1.0,
                        pos=init_p2.tolist(), 
                        euler=(0.0, 0.0, 0.0),
                        decompose_object_error_threshold=0.01,
                        recompute_inertia = True,
                    ),
                    surface=gs.surfaces.Rough(color=(0.9, 0.1, 0.1)),
                    material=gs.materials.Rigid(
                        # needs_coup=True,           # Required for IPC to see this object
                        # coup_restitution=0.5,      # This is your bounciness (0.0 to 1.0)
                        # coup_friction=0.5,         # Add this to control friction in IPC
                        friction=FRICTION2,              # Keep this for general rigid solver consistency
                        rho=RHO2,
                    ),
                )
            except Exception as e:
                print(f"Error: failed to add objects to scene {e}")
                continue
  
            scene.build()
            if args.show_viewer:
                scene.viewer.set_camera_pose(
                    pos=np.array((10.0, 10.0, 10.0)),
                    lookat=np.array((5.0, 5.0, 5.0)),
                )

            # Apply velocities
            object1.set_dofs_velocity(np.concatenate([init_v1, init_w1]))
            object2.set_dofs_velocity(np.concatenate([init_v2, init_w2]))

            # Extract collision meshes for sampling
            mesh_list1 = [geom.get_trimesh() for geom in object1._links[0].geoms]
            mesh1 = trimesh.util.concatenate(mesh_list1)
            mesh_list2 = [geom.get_trimesh() for geom in object2._links[0].geoms]
            mesh2 = trimesh.util.concatenate(mesh_list2)

            # Sample 2048 points for each object
            p1, fi1, bc1 = sample_points_with_face_tracking(mesh1, N * 20)
            p1_torch = torch.tensor(p1, dtype=torch.float32, device=device).contiguous()
            idx1 = fps(p1_torch, ratio=1/20, random_start=True)
            points1, face_indices1, barycentric_coords1 = p1[idx1.cpu()], fi1[idx1.cpu()], bc1[idx1.cpu()]

            p2, fi2, bc2 = sample_points_with_face_tracking(mesh2, N * 20)
            p2_torch = torch.tensor(p2, dtype=torch.float32, device=device).contiguous()
            idx2 = fps(p2_torch, ratio=1/20, random_start=True)
            points2, face_indices2, barycentric_coords2 = p2[idx2.cpu()], fi2[idx2.cpu()], bc2[idx2.cpu()]

            vert_list1 = [object1.get_verts().cpu().numpy()]
            vert_list2 = [object2.get_verts().cpu().numpy()]
            for _ in tqdm(range(144)):
                scene.step()
                scene.visualizer.update_visual_states()
                vert_list1.append(object1.get_verts().cpu().numpy())
                vert_list2.append(object2.get_verts().cpu().numpy())
            print("Finished stepping through simulation!")

            x_list = []
            for f_idx in range(145):
                mesh1.vertices = vert_list1[f_idx]
                tp1 = find_corresponding_points(mesh1, face_indices1, barycentric_coords1)
                
                mesh2.vertices = vert_list2[f_idx]
                tp2 = find_corresponding_points(mesh2, face_indices2, barycentric_coords2)
                
                x_list.append(np.concatenate([tp1, tp2], axis=0))
            print("Finished finding corresponding points!")

            x_list = np.stack(x_list, axis=0)
 
            f = h5py.File(output_path, 'w')
            f.create_dataset('x', data=x_list)
            f.create_dataset('floor_height', data=floor_height)
            f.create_dataset('gravity', data=gravity)
            
            # Save initial conditions
            f.create_dataset('p1', data=init_p1)
            f.create_dataset('p2', data=init_p2)
            f.create_dataset('v1', data=init_v1)
            f.create_dataset('v2', data=init_v2)
            f.create_dataset('w1', data=init_w1)
            f.create_dataset('w2', data=init_w2)
            
            # Save physical properties
            f.create_dataset('diameter1', data=OBJ_SIZE1)
            f.create_dataset('diameter2', data=OBJ_SIZE2)
            f.create_dataset('rho1', data=RHO1)
            f.create_dataset('friction1', data=FRICTION1)
            f.create_dataset('rho2', data=RHO2)
            f.create_dataset('friction2', data=FRICTION2)
            f.create_dataset('floor_friction', data=FLOOR_FRICTION)
            
            f.close()
            print(f"Finished creating {output_path}")
            
            if args.visualization:
                print("Starting visualization...")
                save_pointcloud_video(x_list, [], f'{output_dir}/visualization/{output_idx}.gif', 
                    grid_lim=1.0, vertical_axis='z', floor_height=floor_height, show_trajectory=True, fps=10)
                print("Finished visualizing object collision!")

    if args.show_viewer:
        import threading
        print("Generation finished. Explore the viewer.")
        print("Press Enter in the terminal or close the window to exit.")
        
        exit_event = threading.Event()
        def wait_for_exit():
            input()
            exit_event.set()
            
        thread = threading.Thread(target=wait_for_exit, daemon=True)
        thread.start()
        
        while scene.viewer.is_alive and not exit_event.is_set():
            try:
                scene.step()
            except gs.GenesisException:
                break

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--base-dir', type=str, default='data')
    parser.add_argument('--output-dir', type=str, default='outputs_genesis')
    parser.add_argument('--dataset-type', type=str, default='3d_primitives')
    parser.add_argument('--uid_list', type=str, default='configs/3d_primitives.json')
    parser.add_argument('--start-idx', type=int, default=0)
    parser.add_argument('--end-idx', type=int, default=1)
    parser.add_argument('--start-idx-video', type=int, default=20)
    parser.add_argument('--end-idx-video', type=int, default=21)
    parser.add_argument('--visualization', action='store_true') 
    parser.add_argument('--show-viewer', action='store_true')
    parser.add_argument('--loop', action='store_true')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--scenario', type=int, choices=[1, 2, 3], default=3, help='Collision scenario type (1: no collision, 2: one stationary in air, 3: both moving in air)')
    parser.add_argument('--n_samples', type=int, default=10, help='Number of samples to generate')
    args = parser.parse_args()
    
    if args.loop:
        while True:
            p = Process(target=run_generation, args=(args,))
            p.start()
            p.join()
            
            if p.exitcode != 0:
                print("Generation process crashed, restarting...")
            else:
                print("Generation process finished.")
    else:
        run_generation(args)