# Naive CapApi seed for SimFirst press_shampoo.
# Expected to fail until the iterative-debugging agent rewrites it.
report = {"task": "press_shampoo", "actions": []}
report["actions"].append({"open_gripper": open_gripper().ok})
state = get_robot_state()
pose = state.end_effector_poses["primary"]
above = Pose(
    (float(pose.position[0]), float(pose.position[1]), float(pose.position[2]) - 0.05),
    pose.quaternion_wxyz,
    pose.frame,
)
report["actions"].append({"nudge_down": move_to_pose(above, tolerance=0.05, max_steps=200).ok})
result = report
