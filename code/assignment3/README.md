
## Grasp Loop sim

The main file to run is `franka_grasp_sim.py` which uses `fsm.py` for control using a finite state machine with continuous torque control for each state (HOME, GRASP_OPEN,APPROACH, GRASP_CLOSE, LIFT (unused)). The model is in `robosuite_model.xml`

TODO: `fsm.py` is using a lot of hardcoded variables, try to fix. Example, home joint config is hard-coded for Franka

