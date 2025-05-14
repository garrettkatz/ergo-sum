# Implement the stubs where requested in the comments below
# No imports allowed other than these, do not change these lines
import matplotlib.pyplot as pt
import torch as tr
tr.set_default_dtype(tr.float64) # makes round-off errors much smaller
import rotations as rt
import ergo_jr

def get_frames(joint_info, joint_angles):
    """
    forward kinematics, joint_angles can be batched (b is batch index)
    joint_angles and outputs are torch tensors
    input:
        joint_info has the same format as in ergo_jr.py
        joint_angles[b,j] is jth joint angle in bth batch sample
    returns (locations, orientations) where
        locations[b,i,:] is 3d location of ith joint in base coordinate frame
        orientations[b,i,:] is 4d orientation versor of ith joint in base coordinate frame
    if leading batch dimension is omitted from input, it will also be omitted in output
    i ranges over all "joints" in joint_info
    j ranges over the non-fixed joints only (where axes in joint_info is not None)
    """

    # TEMPLATE-START
    # unpack joint info
    names, parents, translations, rotations, axes = zip(*joint_info)

    # set up dimensions
    batch_dims = joint_angles.shape[:-1]
    num_points = len(joint_info)
    joint_points = tuple(i for (i,a) in enumerate(axes) if a is not None)
    num_joints = len(joint_points)

    # setup children
    children = {i: [] for i in range(num_points)}
    roots = []
    for i, p in enumerate(parents):
        if p == -1: roots.append(i)
        else: children[p].append(i)

    # isolate non-None axes and wrap in tensor
    axes = tr.tensor([axes[i] for i in joint_points])

    # set up local transforms at each point relative to parent
    point_rotations = [rt.identity_versor(batch_dims)] * num_points
    if num_joints > 0:
        axes_rotations = rt.versor_from(axes, joint_angles.unsqueeze(-1))
        for j in range(num_joints):
            point_rotations[joint_points[j]] = axes_rotations[..., j, :]
    point_rotations = tr.stack(point_rotations, dim=-2)
    point_translations = tr.tensor(translations).expand(batch_dims + (-1, -1))

    # vectorize operations across layers of kinematic tree for speed-up
    orientations = tr.tensor(rotations).expand(batch_dims + (-1, -1)).clone()
    points = point_translations.clone() # (..., I, 3)

    # initialize first layer of parents (roots of base) and their children
    P = [r for r in roots for c in children[r]]
    C = [c for r in roots for c in children[r]]
    while len(C) > 0:
        # batch transformations over all children in current layer
        parent_rotations = rt.multiply(orientations[...,P,:], point_rotations[...,P,:])
        points[...,C,:] = points[...,P,:] + rt.rotate(parent_rotations, point_translations[...,C,:])
        orientations[...,C,:] = rt.multiply(parent_rotations, orientations[...,C,:])

        # children become parents for next iteration
        P = [c for c in C for gc in children[c]]
        C = [gc for c in C for gc in children[c]]

    # return results
    return points, orientations
    # TEMPLATE-END

    raise NotImplementedError()

def get_jacobian(joint_info, locations, orientations):
    """
    derivatives of joint locations with respect to joint angles
    joint_info, locations and orientations have the format as in get_frames
    output is jacobian, a torch tensor where
        jacobian[b,i,c,j] is d locations[b,i,c] / d joint_angles[b,j]
        b ranges over the batch dimension
        i ranges over all joints, both fixed and non-fixed
        c ranges over their 3d location coordinates
        j ranges over the non-fixed joints only (where axes in joint_info is not None)
    if leading batch dimension is omitted from input, it will also be omitted in output
    you are meant to compute the jacobian from its formula, without using torch autograd
    """

    # TEMPLATE-START
    # unpack joint info
    names, parents, translations, rotations, axes = zip(*joint_info)

    # set up dimensions
    num_points = len(joint_info)
    joint_points = tuple(i for (i,a) in enumerate(axes) if a is not None)
    num_joints = len(joint_points)

    # work out joint ancestors of each point
    ancestors = {i: [] for i in range(num_points)}
    for i in range(num_points):
        p = parents[i]
        while p > -1:
            if p in joint_points: ancestors[i].append(joint_points.index(p))
            p = parents[p]

    # could be a rigid set of points
    if num_joints == 0: return tr.zeros(locations.shape + (0,))

    # isolate non-None axes and wrap in tensor
    axes = tr.tensor([axes[i] for i in joint_points])

    joint_locations = locations[..., joint_points, :]
    joint_orientations = orientations[..., joint_points, :]
    axes = rt.rotate(joint_orientations, axes) # (..., J, 3)
    jacobian = tr.zeros(locations.shape + (num_joints,)) # (..., I, 3, J)
    for i in range(num_points):
        if len(ancestors[i]) == 0: continue # no joint dependencies
        offset = locations[...,i:i+1,:] - joint_locations[...,ancestors[i],:] # (..., A, 3)
        joint_axes = axes[..., ancestors[i], :] # (..., A, 3)
        tangent = tr.linalg.cross(joint_axes, offset, dim=-1) # (..., A, 3)
        jacobian[...,i,:,ancestors[i]] = tr.transpose(tangent, -2, -1) # (..., 3, A)

    return jacobian
    # TEMPLATE-END

    raise NotImplementedError()


if __name__ == "__main__":

    angles = tr.full((6,), +.2)
    locations, orientations = get_frames(ergo_jr.joint_info, angles)
    jacobian = get_jacobian(ergo_jr.joint_info, locations, orientations)

    jacobian_t = tr.autograd.functional.jacobian(lambda a: get_frames(ergo_jr.joint_info, a)[0], angles)

    print(locations.numpy().round(4))
    print()
    print(orientations.numpy().round(4))
    print()

    print(jacobian.numpy().round(4))
    print()
    print(jacobian_t.numpy().round(4))
    print()

    ax = pt.gcf().add_subplot(projection='3d')
    ax.plot(*locations.numpy().T, 'ko-')
    pt.axis("equal")
    pt.show()

    assert tr.allclose(jacobian, jacobian_t)


