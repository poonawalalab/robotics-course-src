"""
Current version achieves a loop for the franka+2-fingered gripper where it reaches HOME with closed gripper, then opens gripper while going to grab, then home while closing (picking up block), then open gripper (dropping gripper) and go-to-grab etc. 
TODO: proper grasp check logic

The _grasp_close mode has a joint PD control with reference self.grasp_hold and gripper close mode. 
When we transition from APPROACH to GRASP_CLOSE, grasp_hold holds the joint values at transition
When we transition from GRASP_CLOSE TO HOME, grasp_hold holds the home joint angles
"""
from enum import Enum, auto
import mujoco
import numpy as np

class State(Enum):
    HOME = auto()
    GRASP_OPEN = auto()
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

    def transition(self):
        transitions = {
            State.HOME:    State.GRASP_OPEN,
            State.GRASP_OPEN:    State.APPROACH,
            State.APPROACH:    State.GRASP_CLOSE,
            State.GRASP_CLOSE: State.HOME, #loops
            ## to terminate after lifting:
            # State.GRASP_CLOSE: State.LIFT,
            # State.LIFT: State.LIFT, ## terminal state
        }
        self.state = transitions[self.state]

    def update(self, model, data):
        if self.state == State.HOME:
            self._home(model, data)
        elif self.state == State.GRASP_OPEN:
            self._grasp_open(model, data)
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
        if  self._near_home(model,data):
            print("transition to GRASP OPEN at ",data.time)
            self.transition()

    def _grasp_open(self, model, data):
        # open gripper fingers
        self._gripper_open(model,data) ## 
        if data.qpos[model.nu-2] > 0.039:
            print("transition to APPROACH at ",data.time)
            self.transition()

    def _approach(self, model, data):
        # set arm joint targets, check if near enough
        self._diff_ik_control(model,data)
        if self._near_object(data):
            print("transition to GRASP CLOSE at ",data.time)
            self.grasp_hold = data.qpos[:(model.nu-2)].copy() ## hold joint values corresponding to IK solution
            self.transition()

    def _grasp_close(self, model, data):
        # close gripper fingers
        self._gripper_close(model,data) ## 
        if self._grasp_stable(model,data):
            print("transition to HOME at ",data.time)
            self.grasp_hold = self.q_home.copy()
            self.transition()

    ## Unused LIFT

    def _lift(self, model, data):
        # move arm upward
        self._gripper_close(model,data)
        pass

    # --- transition conditions ---

    def _near_home(self, model,data):
        err =  np.linalg.norm(self.q_home - data.qpos[:(model.nu-2)]) 
        return err < 0.15

    def _near_object(self, data):
        # e.g. check distance between end effector and object body
        dx = data.body("cube_main").xpos - data.site(self.site_id).xpos
        return np.linalg.norm(dx) < 0.005

    def _grasp_stable(self,model, data):
        # e.g. check contact forces or finger position error is small
        # Here, we check if fingers have stopped moving, excluding start of 0.04
        return np.linalg.norm(data.qvel[(model.nu-2):model.nu]) < 0.001 and data.qpos[7] < 0.037

    # --- utils ---

    def _gripper_open(self,model,data):
        Gq = self._gravity_compensation(model,data)
        data.ctrl[:(model.nu-2)] =  4* (self.grasp_hold - data.qpos[:(model.nu-2)]) - 3*( data.qvel[:(model.nu-2)]) + Gq[:(model.nu-2)] # original torque based  control grav compensation + P velocity
        data.ctrl[(model.nu-2):]= 100*(np.array([0.04,-0.04]) - data.qpos[(model.nu-2):model.nu]) - 20*data.qvel[(model.nu-2):model.nu]
        # self.print_contacts(model,data)

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
        dq = jacpos.T @ np.linalg.solve(jacpos @ jacpos.T + diag3, twist[:3])
        # null-space control biasing to home joint angles
        dq_target=np.zeros(model.nv)
        dq_target[:(model.nu-2)] = self.q_home - data.qpos[:(model.nu-2)]
        dq += (eye - np.linalg.pinv(jacpos) @ jacpos) @ ( dq_target)

        # Clamp maximum joint velocity.
        dq_abs_max = np.abs(dq).max()
        if dq_abs_max > self.max_angvel:
            dq *= self.max_angvel / dq_abs_max

        # Set the control signal and step the simulation.
        Gq = self._gravity_compensation(model,data)

        data.ctrl[:(model.nu-2)] = 10*( dq[:(model.nu-2)] - data.qvel[:(model.nu-2)]) + Gq[:(model.nu-2)] # original torque based  control grav compensation + P velocity
        data.ctrl[(model.nu-2):]= 100*(np.array([0.04,-0.04]) - data.qpos[(model.nu-2):model.nu]) - 20*data.qvel[(model.nu-2):model.nu]
        # data.ctrl[6:]= 10*(np.array([0.04,0.04])+0.0*np.array([0.02,0.02])*np.sin(data.time) - data.qpos[6:8]) - 3*data.qvel[6:8]


    def print_contacts(self,model,data):
        print("\nn contacts:", data.ncon)
        for i in range(data.ncon):
            con = data.contact[i]
            # ID of geoms in contact
            print("geoms:", model.geom(con.geom1).name, model.geom( con.geom2).name)
            if ( model.geom( con.geom2).name == "gripper0_right_finger2_pad_collision" or  model.geom( con.geom1).name == "gripper0_right_finger2_pad_collision" or  model.geom( con.geom2).name == "gripper0_right_finger1_pad_collision" or  model.geom( con.geom1).name == "gripper0_right_finger1_pad_collision"):
                forcetorque = np.zeros(6)
                mujoco.mj_contactForce(model, data, i, forcetorque)
                contact_force_local = forcetorque[:3] # [fx, fy, fz] in contact frame
                print("force: ",contact_force_local)


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
