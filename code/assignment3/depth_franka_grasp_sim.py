import mujoco
import mujoco.viewer
import time 
import argparse
import multiprocessing as mp
import cv2
import numpy as np

depth_state = {"depth_m": None}
rgb_state   = {"rgb": None}
zaxis_vector = np.array([0,0,1.0]);
max_angvel = 0.785
q_home = np.array([0,0,0,-1.57079,0,1.57079,0.7853])
Kpos: float = 0.95
Kori: float = 0.95
integration_dt = 0.1
damping: float = 1e-4
grasp_hold = q_home.copy()
grasp_x = np.array([0,0,0.82])


def _depth_viewer(queue: mp.Queue):
    """Subprocess: receives (H,W) depth arrays and shows them with OpenCV."""
    cv2.namedWindow("eye_in_hand rgb",   cv2.WINDOW_NORMAL)
    cv2.namedWindow("eye_in_hand depth", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("eye_in_hand rgb",   512, 512)
    cv2.resizeWindow("eye_in_hand depth", 512, 512)
    cv2.moveWindow("eye_in_hand rgb",     0,   0)
    cv2.moveWindow("eye_in_hand depth", 520,   0)
    while True:
        frame = queue.get()
        if frame is None:
            break
        depth = frame["depth"]
        rgb = frame["rgb"]
        near = depth[depth > 0].min() if (depth > 0).any() else 0.1
        far  = depth.max() if depth.max() > 0 else 1.0
        norm = np.clip((depth - near) / (far - near + 1e-6), 0, 1)
        colored = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
        cv2.imshow("eye_in_hand depth", colored)
        cv2.imshow("eye_in_hand rgb", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)



def convert_vec_to_quat(v):
    ## The x and y axis may not change a lot based on this quaternion
    ## To fix, you want to find the rotation based on current quaternion as start, not identity orientation
    ## For safety, assert that v is a numpy array of dimension 3 
    v = v / (np.linalg.norm(v) + 0.001)
    res = np.zeros(3)
    mujoco.mju_cross(res, zaxis_vector, v)
    sin_angle = np.linalg.norm(res)
    cos_angle = v[2]  # dot(zaxis, v)
    if sin_angle < 1e-7:
      if cos_angle > 0:
          return np.array([1.0, 0.0, 0.0, 0.0])   # already aligned
      else:
          return np.array([0.0, 1.0, 0.0, 0.0])   # 180° flip around x
    quat = np.array([1.0 + cos_angle, res[0], res[1], res[2]])
    mujoco.mju_normalize4(quat)
    return quat     


def convert_vec_to_err_quat(curr_quat,v):
    ## For safety, assert that v is a numpy array of dimension 3 
    v = v / (np.linalg.norm(v)+0.001)
    c = np.zeros(3)
    w = curr_quat[0]
    x = curr_quat[1]
    y = curr_quat[2]
    z = curr_quat[3]
    c[0] = 2*x*z+2*w*y
    c[1] = 2*y*z-2*w*x
    c[2] = w*w - x*x-y*y+z*z
    res = np.zeros(3);
    mujoco.mju_cross(res,c,v) 
    sin_angle = np.linalg.norm(res)
    cos_angle = c @ v  # dot(zaxis, v)
    if sin_angle < 1e-7:
      if cos_angle > 0:
          return np.array([1.0, 0.0, 0.0, 0.0])   # already aligned
      else:
          return np.array([0.0, 1.0, 0.0, 0.0])   # 180° flip around x
    quat = np.array([1.0 + cos_angle, res[0], res[1], res[2]])
    ## if costheta = a^T v with a as the unit-z axis (zaxis_vector)
    ## and c = a x v (cross product)  . Closed form is possible here
    ## Then a quaternion with z along v is (unnormalized):
    # 3. Always normalize quaternions to ensure they are valid unit vectors
    mujoco.mju_normalize4(quat)
    return quat
    

def find_cube_pixel(rgb):
    """
    Return (row, col) centroid of the cube in the raw (un-flipped) image,
    or None if not found.

    The Lift cube is red/orange: high R, low G, low B.
    Tune thresholds here if detection is unreliable.
    """
    r = rgb[:, :, 0].astype(float)
    g = rgb[:, :, 1].astype(float)
    b = rgb[:, :, 2].astype(float)
    mask = (r > 150) & (r > 1.5 * g) & (r > 1.5 * b)
    if mask.sum() < 5:
        return None
    rows, cols = np.where(mask)
    return int(rows.mean()), int(cols.mean())

def find_cube_depth(rgb,depth_m):
    """
    Return (row, col) centroid of the cube in the raw (un-flipped) image,
    or None if not found.

    The Lift cube is red/orange: high R, low G, low B.
    Tune thresholds here if detection is unreliable.
    """
    r = rgb[:, :, 0].astype(float)
    g = rgb[:, :, 1].astype(float)
    b = rgb[:, :, 2].astype(float)
    mask = (r > 150) & (r > 1.5 * g) & (r > 1.5 * b)
    if mask.sum() < 5:
        return None
    inv_depth = np.where(mask, 1.0 / depth_m, 0.0)
    rows, cols = np.where(mask)
    row, col = int(rows.mean()), int(cols.mean())
    cube_depth = inv_depth[row, col]   # real metres to cube centre
    return int(rows.mean()), int(cols.mean()), cube_depth

def depthcontroller(model, data):
    site_id = model.site("gripper0_right_grip_site").id
    depth = depth_state["depth_m"]
    if depth is not None:
        # use depth for control here
        pass
    else:
        print("no depth")
    rgb = rgb_state["rgb"]
    if rgb is not None:
        # rgb is (H, W, 3) uint8, same orientation as depth_m (pre-flipud)
        pass
    else:
        print("no rgb")
    ## Point to object 
    jac = np.zeros((6, model.nv))
    jacpos_prev = np.zeros((3, model.nv))   # previous-step Jacobian for finite-difference J̇
    jacpos = np.zeros((3, model.nv))
    jacori = np.zeros((3, model.nv))
    diag =  damping * np.eye(model.nv)
    diag3 = damping * np.eye(3)
    diag6 = damping * np.eye(6)
    eye = np.eye(model.nv)
    twist = np.zeros(6)
    site_quat = np.zeros(4)
    site_quat_conj = np.zeros(4)
    error_quat = np.zeros(4)
    dx = data.body("cube_main").xpos - data.site(site_id).xpos
    ## Skip the reaching velocity
    twist[:3] = 0*Kpos * dx / integration_dt
    mujoco.mju_mat2Quat(site_quat, data.site(site_id).xmat)
    mujoco.mju_negQuat(site_quat_conj, site_quat)
    mujoco.mju_mulQuat(error_quat, convert_vec_to_quat(dx) , site_quat_conj)
    # mujoco.mju_quat2Vel(twist[3:], error_quat, 1.0)
    mujoco.mju_quat2Vel(twist[3:], convert_vec_to_err_quat(site_quat,dx), 1.0)
    twist[3:] *= Kori / integration_dt   
    # Jacobian.
    mujoco.mj_jacSite(model, data, jac[:3], jac[3:], site_id)
    mujoco.mj_jacSite(model, data, jacpos, jacori, site_id)

    # Damped least squares.
    # dq = jacpos.T @ np.linalg.solve(jacpos @ jacpos.T + diag3, twist[:3])
    dq = jacori.T @ np.linalg.solve(jacori @ jacori.T + diag3, twist[3:])
    # dq = jac.T @ np.linalg.solve(jac @ jac.T + diag6, twist)
    # print("twist: ",dq)

    # null-space control biasing to home joint angles
    dq_target=np.zeros(model.nv)
    dq_target[:(model.nu-2)] = q_home - data.qpos[:(model.nu-2)]
    dq += (eye - np.linalg.pinv(jacori) @ jacpos) @ ( dq_target)
    # print("add home: ",dq)

    # Clamp maximum joint velocity.
    dq_abs_max = np.abs(dq).max()
    if dq_abs_max > max_angvel:
        dq *= max_angvel / dq_abs_max

    ## Get gravity term for compensation: 
    nv = model.nv
    qd_saved = data.qvel[:nv].copy()
    data.qvel[:nv] = 0.0
    mujoco.set_mjcb_control(None) # deactivate callback to prevent recursion on forward
    mujoco.mj_forward(model, data)
    Gq = data.qfrc_bias[:nv].copy()
    data.qvel[:nv] = qd_saved
    mujoco.mj_forward(model, data)   # restore kinematics
    mujoco.set_mjcb_control(depthcontroller) #restore callback

    if rgb is not None:
        pixel = find_cube_pixel(rgb)
        if pixel is not None:
            row, col = pixel
            print(f"row: {row} col: {col}")
            if depth is not None:
                row, col, inv_d = find_cube_depth(rgb,depth)
                print(f" depth: {depth[row,col]} inv depth: {inv_d}")


    ## PD control to target position 
    #data.ctrl[:(model.nu-2)] =  4* (q_home - data.qpos[:(model.nu-2)]) - 3*( data.qvel[:(model.nu-2)]) + Gq[:(model.nu-2)] # original torque based  control grav compensation + P velocity

    ## P control to target velocity
    data.ctrl[:(model.nu-2)] = 10*( dq[:(model.nu-2)] - data.qvel[:(model.nu-2)]) + Gq[:(model.nu-2)] # original torque based  control grav compensation + P velocity
    ## P control to finger position:
    data.ctrl[(model.nu-2):]= 100*(np.array([0.04,-0.04]) - data.qpos[(model.nu-2):model.nu]) - 20*data.qvel[(model.nu-2):model.nu]


def mycallback(model, data):
    ## Currently a joint-space PD controller targeting open@home 
    ## Do all this to get gravity term:
    nv = model.nv
    qd_saved = data.qvel[:nv].copy()
    data.qvel[:nv] = 0.0
    mujoco.set_mjcb_control(None) # deactivate callback to prevent recursion on forward
    mujoco.mj_forward(model, data)
    Gq = data.qfrc_bias[:nv].copy()
    data.qvel[:nv] = qd_saved
    mujoco.mj_forward(model, data)   # restore kinematics
    mujoco.set_mjcb_control(mycallback) #restore callback
    data.ctrl[:(model.nu-2)] =  4* (q_home - data.qpos[:(model.nu-2)]) - 3*( data.qvel[:(model.nu-2)]) + Gq[:(model.nu-2)] # original torque based  control grav compensation + P velocity
    data.ctrl[(model.nu-2):]= 100*(np.array([0.04,-0.04]) - data.qpos[(model.nu-2):model.nu]) - 20*data.qvel[(model.nu-2):model.nu]

CAM  = "robot0_eye_in_hand"
H, W = 128, 128
DEPTH_EVERY_N = 10   # render depth every N sim steps to avoid slowing things down
q_home = np.array([0,0,0,-1.57079,0,1.57079,0.7853])
# Load a model from a file (won't do this here to avoid external file calls)
def main(args):
    depth_queue = mp.Queue(maxsize=1)
    depth_proc  = mp.Process(target=_depth_viewer, args=(depth_queue,), daemon=True)
    depth_proc.start()
    model = mujoco.MjModel.from_xml_path(args.filename)

    data = mujoco.MjData(model)
    print("actuators: ", model.nu)
    # for i in range(0,model.nq):
    #     data.qpos[i]=0.0

    renderer = mujoco.Renderer(model, height=H, width=W)
    renderer.enable_depth_rendering()
    rgb_renderer = mujoco.Renderer(model, height=H, width=W)  # RGB by default

    # mujoco.set_mjcb_control(mycallback)
    mujoco.set_mjcb_control(depthcontroller)


    # Create a viewer. Use `mjpython` on `macOS` due to use of non-blocking `launch_passive` instead of blocking `launch`
    viewer = mujoco.viewer.launch_passive(model, data, 
            show_left_ui=False,
            show_right_ui=False,
    )
    if args.frames:
        # viewer.opt.frame = mujoco.mjtFrame.mjFRAME_BODY # Enables visualization of body frames
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE # Enables visualization of body frames
        viewer.opt.sitegroup[:]=1
        viewer.opt.sitegroup[1]=1
        # You can also enable other visual aids like joint axes or contact forces
    # viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True

    step = 0
    while viewer.is_running():
        mujoco.mj_step(model, data)
        if step % DEPTH_EVERY_N == 0:
            renderer.update_scene(data, camera=CAM)
            depth = renderer.render()          # (H, W), normalised [0,1]
            extent = model.stat.extent
            far    = model.vis.map.zfar  * extent
            near_z = model.vis.map.znear * extent
            depth_m = near_z / (1.0 - depth * (1.0 - near_z / far))  # real metres
            depth_state["depth_m"] = depth_m          # write
            rgb_renderer.update_scene(data, camera=CAM)
            rgb_state["rgb"] = rgb_renderer.render()   # (H, W, 3) uint8 RGB
            if not depth_queue.full():
                depth_queue.put_nowait({"depth": np.flipud(depth_m).copy(),
                           "rgb":   np.flipud(rgb_state["rgb"]).copy()})
        step += 1
        viewer.sync()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a path planning solution")
    parser.add_argument("--filename", type=str, default = "robosuite_model.xml", help="xml file name to render")
    parser.add_argument("--frames",type=bool, default = False)
    args = parser.parse_args()
    main(args)
