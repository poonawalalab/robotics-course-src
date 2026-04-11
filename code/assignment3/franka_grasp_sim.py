import mujoco
import numpy as np
import time
from fsm import State, ArmFSM

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

def main():
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

        while viewer.is_running():
            step_start = time.time()

            mujoco.mj_step(model, data)

            viewer.sync()
            time_until_next_step = dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)



if __name__ == "__main__":
    main()
