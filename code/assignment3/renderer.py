import mujoco
import mujoco.viewer
import time 
import argparse

# Load a model from a file (won't do this here to avoid external file calls)
def main(args):
    model = mujoco.MjModel.from_xml_path(args.filename)

    data = mujoco.MjData(model)
    print("actuators: ", model.nu)
    # for i in range(0,model.nq):
    #     data.qpos[i]=0.0


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

    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a path planning solution")
    parser.add_argument("filename", type=str, default = "franka_scene.xml", help="xml file name to render")
    parser.add_argument("--frames",type=bool, default = False)
    args = parser.parse_args()
    main(args)
