"""
The kinematic parameters for the ergo jr robot

joint_info[i] = (joint_name, parent_index, translation, orientation, axis) for ith joint
joint_name: string identifier for joint
parent_index: index in joint_info of parent joint (-1 is base)
translation: (3,) translation vector in parent joint's local frame
orientation: (4,) orientation versor in parent joint's local frame
axis: (3,) rotation axis vector (should be unit length) in ith joint's own local frame (None for fixed "joints")

"""
joint_info = [
    ('m1', -1, (0.0, 0.0, 0.0327993216120967), (1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    ('m2', 0, (0.0, 0.0, 0.0240006783879033), (0.7071067811865462, 0.0, -0.7071067811865488, 0.0), (0.0, 0.0, -1.0)),
    ('m3', 1, (0.054, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), (0.0, 0.0, -1.0)),
    ('m4', 2, (0.045, 0.0, 0.0), (-0.7071067811865462, 0.0, 0.7071067811865488, 0.0), (0.0, 0.0, -1.0)),
    ('m5', 3, (0.0, -0.048, 0.0), (0.7071067811865462, 0.0, -0.7071067811865488, 0.0), (0.0, 0.0, 1.0)),
    ('t7f', 4, (0.0, -0.125, 0.0155), (1.0, 0.0, 0.0, 0.0), None), # fixed finger tip
    ('m6', 4, (0.0, -0.058, 0.0), (0.7071067811865462, 0.0, -0.7071067811865488, 0.0), (0.0, 0.0, -1.0)), # gripper motor
    ('t7m', 6, (-0.0155, -0.0675, 0.0), (1.0, 0.0, 0.0, 0.0), None), # movable finger tip
]

