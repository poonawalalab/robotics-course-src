import mujoco
import numpy as np
import time
import multiprocessing as mp
import cv2
from fsm import State, ArmFSM


def _depth_viewer(queue: mp.Queue):
    """Subprocess: receives (H,W) depth arrays and shows them with OpenCV."""
    while True:
        depth = queue.get()
        if depth is None:
            break
        near = depth[depth > 0].min() if (depth > 0).any() else 0.1
        far  = depth.max() if depth.max() > 0 else 1.0
        norm = np.clip((depth - near) / (far - near + 1e-6), 0, 1)
        colored = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
        cv2.imshow("eye_in_hand depth", colored)
        cv2.waitKey(1)

# Integration timestep in seconds. This corresponds to the amount of time the joint
# velocities will be integrated for to obtain the desired joint positions.
integration_dt: float = 0.1

# Damping term for the pseudoinverse. This is used to prevent joint velocities from
# becoming too large when the Jacobian is close to singular.
damping: float = 1e-4
gravity_compensation: bool = True
dt: float = 0.002

# Gains for the twist computation. These should be between 0 and 1. 0 means no
# movement, 1 means move the end-effector to the target in one integration step.

CAM  = "robot0_eye_in_hand"
H, W = 128, 128
DEPTH_EVERY_N = 10   # render depth every N sim steps to avoid slowing things down


def main():
    depth_queue = mp.Queue(maxsize=1)
    depth_proc  = mp.Process(target=_depth_viewer, args=(depth_queue,), daemon=True)
    depth_proc.start()

    model = mujoco.MjModel.from_xml_path("robosuite_model.xml")
    data = mujoco.MjData(model)
    # data.qpos[:8] = np.array([0, 1.67045757, 1.71305828,-1.26869675,-0.12834041 ,0,0.04,0.04])
    # Enable gravity compensation. Set to 0.0 to disable.
    model.body_gravcomp[:] = float(gravity_compensation)
    model.opt.timestep = dt
    fsm = ArmFSM(model,data)
    # data.qpos[:(model.nu-2)] = fsm.q_home
    # mujoco.mj_forward(model,data)

    def mycallback(model, data):
        fsm.update(model, data)

    fsm.callback = mycallback
    mujoco.set_mjcb_control(mycallback)

    renderer = mujoco.Renderer(model, height=H, width=W)
    renderer.enable_depth_rendering()

    with mujoco.viewer.launch_passive(
        model=model,
        data=data,
        show_left_ui=False,
        show_right_ui=False,
    ) as viewer:
        # Reset the simulation.
        # viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE # Enables visualization of body frames
        # mujoco.mj_resetDataKeyframe(model, data, key_id)
        # data.qpos[0] = 0.5
        # data.qpos[4] = 0.7 

        # Reset the free camera.
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        # Enable site frame visualization.
        # viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE

        step = 0
        while viewer.is_running():
            step_start = time.time()

            mujoco.mj_step(model, data)

            if step % DEPTH_EVERY_N == 0:
                renderer.update_scene(data, camera=CAM)
                depth = renderer.render()          # (H, W), normalised [0,1]
                extent = model.stat.extent
                far    = model.vis.map.zfar  * extent
                near_z = model.vis.map.znear * extent
                depth_m = near_z / (1.0 - depth * (1.0 - near_z / far))  # real metres
                if not depth_queue.full():
                    depth_queue.put_nowait(np.flipud(depth_m).copy())
            step += 1

            viewer.sync()
            time_until_next_step = dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

        depth_queue.put(None)   # shut down display subprocess
        depth_proc.join()



if __name__ == "__main__":
    main()
