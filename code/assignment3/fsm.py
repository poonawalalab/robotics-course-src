"""
Current version achieves a loop for the franka+2-fingered gripper where it reaches HOME with closed gripper, then opens gripper while going to grab, then home while closing (picking up block), then open gripper (dropping gripper) and go-to-grab etc. 
TODO: proper grasp check logic

"""
from enum import Enum, auto
import mujoco
import numpy as np

class State(Enum):
    HOME = auto()
    APPROACH = auto()
    GRASP_CLOSE = auto()
    LIFT = auto()

class ArmFSM:
    def __init__(self,model,data):
        self.state = State.HOME
        # End-effector site we wish to control.
        self.site_id = model.site("grasp_site").id
        self.mocap_id = model.body("target").mocapid[0]
        # self.q_home = np.array([-0.05982957, 1.67045757, 1.71305828,-1.56869675,-0.12834041 ,0]) #yam
        self.q_home = np.array([0,0,0,-1.57079,0,1.57079,0.7853])
        self.Kpos: float = 0.95
        self.Kori: float = 0.95
        self.integration_dt = 0.1
        self.damping: float = 1e-4
        self.grasp_hold = self.q_home.copy()
        self.grasp_x = np.array([0,0,0.82])

        # Whether to enable gravity compensation.
        self.gravity_compensation: bool = True

        # Simulation timestep in seconds.
        self.dt: float = 0.002

        # Nullspace P gain.
        self.Kn = np.asarray([10.0, 10.0, 10.0, 10.0, 5.0, 5.0, 5.0,0.0,0.0,0.0,0.0,0.0,0.0])
        self.max_angvel = 0.785

    def print_contacts(self,model,data):
        print("\nn contacts:", data.ncon)
        # print("gripper pos:", data.qpos[6:8])
        for i in range(data.ncon):
            con = data.contact[i]
            # Position of contact
            # ID of geoms in contact
            if ( model.geom( con.geom2).name == "tip_left" or  model.geom( con.geom1).name == "tip_left" or  model.geom( con.geom2).name == "tip_right" or  model.geom( con.geom1).name == "tip_right"):
                print("geoms:", model.geom(con.geom1).name, model.geom( con.geom2).name)
                forcetorque = np.zeros(6)
                # i is the index of the contact in data.contact
                mujoco.mj_contactForce(model, data, i, forcetorque)
                contact_force_local = forcetorque[:3] # [fx, fy, fz] in contact frame
                print("force: ",contact_force_local)


    def transition(self):
        transitions = {
            State.HOME:    State.APPROACH,
            State.APPROACH:    State.GRASP_CLOSE,
            State.GRASP_CLOSE: State.HOME, #loops
            ## to terminate after lifting:
            # State.GRASP_CLOSE: State.LIFT,
            # State.LIFT: State.LIFT, ## terminal state
        }
        self.state = transitions[self.state]

    def _gripper_close(self,model,data):
        Gq = self._gravity_compensation(model,data)
        data.ctrl[:(model.nu-2)] =  4* (self.grasp_hold - data.qpos[:(model.nu-2)]) - 3*( data.qvel[:(model.nu-2)]) + Gq[:(model.nu-2)] # original torque based  control grav compensation + P velocity
        data.ctrl[(model.nu-2):]= 100*(np.array([0.001,0.001]) - data.qpos[(model.nu-2):model.nu]) - 20*data.qvel[(model.nu-2):model.nu]
        # self.print_contacts(model,data)

    def _diff_ik_control(self,model,data):
        # print("mocap: ",data.mocap_pos[self.mocap_id])
        # for i in range(model.nbody):
        #     print(i)
        # print(data.body("cube_main").xpos)
        jac = np.zeros((6, model.nv))
        jacpos_prev = np.zeros((3, model.nv))   # previous-step Jacobian for finite-difference J̇
        jacpos = np.zeros((3, model.nv))
        jacori = np.zeros((3, model.nv))
        diag = self.damping * np.eye(model.nv)
        diag3 = self.damping * np.eye(3)
        eye = np.eye(model.nv)
        twist = np.zeros(6)
        site_quat = np.zeros(4)
        site_quat_conj = np.zeros(4)
        error_quat = np.zeros(4)
        dx = self.grasp_x - data.site(self.site_id).xpos # fixed grasp
        dx = data.body("cube_main").xpos - data.site(self.site_id).xpos
        twist[:3] = self.Kpos * dx / self.integration_dt
        mujoco.mju_mat2Quat(site_quat, data.site(self.site_id).xmat)
        mujoco.mju_negQuat(site_quat_conj, site_quat)
        mujoco.mju_mulQuat(error_quat, data.mocap_quat[self.mocap_id], site_quat_conj)
        mujoco.mju_quat2Vel(twist[3:], error_quat, 1.0)
        twist[3:] *= self.Kori / self.integration_dt   
        # Jacobian.
        mujoco.mj_jacSite(model, data, jac[:3], jac[3:], self.site_id)
        mujoco.mj_jacSite(model, data, jacpos, jacori, self.site_id)

        # Damped least squares.
        # dq = np.linalg.solve(jac.T @ jac + diag, jac.T @ twist)
        dq = jacpos.T @ np.linalg.solve(jacpos @ jacpos.T + diag3, twist[:3])
        dq_target=np.zeros(model.nv)
        dq_target[:(model.nu-2)] = self.q_home - data.qpos[:(model.nu-2)]
        dq += (eye - np.linalg.pinv(jacpos) @ jacpos) @ ( dq_target)

        # Clamp maximum joint velocity.
        dq_abs_max = np.abs(dq).max()
        if dq_abs_max > self.max_angvel:
            dq *= self.max_angvel / dq_abs_max

        # # Integrate joint velocities to obtain joint positions.
        q = data.qpos.copy()  # Note the copy here is important.
        mujoco.mj_integratePos(model, q, dq, self.integration_dt)
        # np.clip(q, *model.jnt_range.T, out=q)

        # Set the control signal and step the simulation.
        Gq = self._gravity_compensation(model,data)

        data.ctrl[:(model.nu-2)] = 10*( dq[:(model.nu-2)] - data.qvel[:(model.nu-2)]) + Gq[:(model.nu-2)] # original torque based  control grav compensation + P velocity
        data.ctrl[(model.nu-2):]= 100*(np.array([0.04,-0.04]) - data.qpos[(model.nu-2):model.nu]) - 20*data.qvel[(model.nu-2):model.nu]
        # data.ctrl[6:]= 10*(np.array([0.04,0.04])+0.0*np.array([0.02,0.02])*np.sin(data.time) - data.qpos[6:8]) - 3*data.qvel[6:8]

    def _gravity_compensation(self,model,data):
        # Gravity compensation: evaluate G(q) with q̇ = 0
        nv = model.nv
        qd_saved = data.qvel[:nv].copy()
        data.qvel[:nv] = 0.0
        mujoco.set_mjcb_control(None) # deactivate callback to prevent recursion
        mujoco.mj_forward(model, data)
        Gq = data.qfrc_bias[:nv].copy()
        data.qvel[:nv] = qd_saved
        mujoco.mj_forward(model, data)   # restore kinematics
        mujoco.set_mjcb_control(self.callback) #restore callback
        return Gq

    def update(self, model, data):
        if self.state == State.HOME:
            self._home(model, data)
        if self.state == State.APPROACH:
            self._approach(model, data)
        elif self.state == State.GRASP_CLOSE:
            self._grasp_close(model, data)
        elif self.state == State.LIFT:
            self._lift(model, data)

    # --- per-state logic ---

    def _home(self, model, data):
        # set arm joint targets, check if near enough
        self._gripper_close(model,data)
        err = np.linalg.norm(self.q_home - data.qpos[:(model.nu-2)])
        # print("error in joint position: " , err)
        if  err < 0.15:
            print("transition from HOME:")
            self.transition()

    def _approach(self, model, data):
        # set arm joint targets, check if near enough
        self._diff_ik_control(model,data)
        if self._near_object(data):
            self.grasp_hold = data.qpos[:(model.nu-2)].copy()
            print("hold:",self.grasp_hold)
            self.transition()

    def _grasp_close(self, model, data):
        # close gripper fingers
        self._gripper_close(model,data)
        if self._grasp_stable(data):
            print("transition to HOME:",self.grasp_hold)
            self.transition()

    def _grasp_close(self, model, data):
        # close gripper fingers
        self._gripper_close(model,data)
        if self._grasp_stable(data):
            print("transition to LIFT:")
            self.grasp_hold = self.q_home.copy()
            self.transition()

    def _lift(self, model, data):
        # move arm upward
        self._gripper_close(model,data)
        pass

    # --- transition conditions ---

    def _near_object(self, data):
        # e.g. check distance between end effector and object body
        dx = data.body("cube_main").xpos - data.site(self.site_id).xpos
        # print( " error from home" ,np.linalg.norm(data.qpos[:6]-self.q_home))
        # print( " error from home" ,np.linalg.norm(dx))
        return np.linalg.norm(dx) < 0.005

    def _grasp_stable(self, data):
        # e.g. check contact forces or finger position error is small
        # print("time (transition @ 15):", data.time)
        return data.time > 15.0
