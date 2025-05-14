import numpy as np

def identity_versor(batch_dims=()):
    """
    Returns the identity versor representing no rotation
    If requested, duplicates the versor across any leading batch dimensions
    batch_dims[k] is the size of the kth batch dimension
    if batch_dims is an int, it is the size of a single leading batch dimension
    """
    # if type(batch_dims) == int: batch_dims = (batch_dims,)
    # return np.ones(batch_dims)[...,None] * np.array([1.,0.,0.,0.])
    return np.array([1.,0,0,0])

def versor_from(axis, angle, normalize=False):
    """
    Return versor representation of rotation by angle about axis
    axis and angle are arrays with shapes (..., 3) and (..., 1)
    where ... is 0 or more leading batch dimensions
    returned versor batch should have shape (..., 4)
    If normalize is True, each axis must be scaled to unit norm
    axis or angle can be missing batch dimensions in which case they are broadcasted
    angle can also be a float in which case it is promoted to array and broadcasted
    """

    # promote scalar angles
    if type(angle) != np.array: angle = np.array(angle)

    # normalize axis if requested
    if normalize: axis /= np.linalg.norm(axis, axis=-1, keepdims=True)

    # broadcast axis and angle batch dimensions to (..., 3) and (..., 1)
    if len(angle.shape) == 0 or angle.shape[-1] > 1: angle = angle[...,None]
    # shape = np.broadcast_shapes(axis.shape, angle.shape)
    # axis = axis.expand(shape)
    # angle = angle.expand(shape[:-1] + (1,))

    # build the versors
    real = np.cos(angle / 2)
    imag = np.sin(angle / 2) * axis
    result = np.concatenate([real, imag], axis=-1)
    return result

# constant coefficient array for converting versor q to rotation matrix M
# _r[i,j] is a 4x4 coefficient for quadratic function q.T @ _r[i,j] @ q
# Programmatically that means
# M[i,j] = (_r[i,j] * q[None, None, None, :] * q[None, None, :, None]).sum(dim=(-2,-1))
_r = np.array([
    [[[1,0, 0,0],[0,1,0,0],[ 0,0,-1,0],[0,0,0,-1]], [[0,0,0,-1],[0, 0,1,0],[0,1,0,0],[-1,0,0, 0]], [[0, 0,1,0],[ 0, 0,0,1],[1,0, 0,0],[0,1,0,0]]],
    [[[0,0, 0,1],[0,0,1,0],[ 0,1, 0,0],[1,0,0, 0]], [[1,0,0, 0],[0,-1,0,0],[0,0,1,0],[ 0,0,0,-1]], [[0,-1,0,0],[-1, 0,0,0],[0,0, 0,1],[0,0,1,0]]],
    [[[0,0,-1,0],[0,0,0,1],[-1,0, 0,0],[0,1,0, 0]], [[0,1,0, 0],[1, 0,0,0],[0,0,0,1],[ 0,0,1, 0]], [[1, 0,0,0],[ 0,-1,0,0],[0,0,-1,0],[0,0,0,1]]],
])

def matrix_from(versor):
    """
    Returns the rotation matrix corresponding to the given versor
    versor is shape (..., 4) where ... are 0 or more leading batch dimensions
    returns mats, a batch of rotation matrices with shape (..., 3, 3)
    mats[b,:,k] is rotation versor[b] applied to kth coordinate axis
    """
    mats = (_r * versor[...,None,None,:,None] * versor[...,None,None,None,:]).sum(axis=(-2,-1)) # (..., 3, 3)
    return mats

def rotate(q, v):
    """
    Returns v_rot, the result of applying the rotation represented by versor q to the 3D vector v
    q has shape (..., 4) while v and v_rot each have shape (..., 3),
    where ... are 0 or more leading batch dimensions
    q or v can be missing batch dimensions, in which case they are broadcasted
    """

    real, imag = q[..., :1], q[..., 1:]
    real, imag, v = np.broadcast_arrays(real, imag, v)
    return v + 2 * np.linalg.cross(imag, (np.linalg.cross(imag, v) + real*v)) # cross defaults across last dimension


# coefficient array for multiply
# q3[i] = (_m[i] * q1[None,:,None] * q2[None,None,:]).sum(dim=(-2, -1))
_m = np.array([
    [[1,0,0,0],[0,-1,0, 0],[0, 0,-1,0],[0,0, 0,-1]],
    [[0,1,0,0],[1, 0,0, 0],[0, 0, 0,1],[0,0,-1, 0]],
    [[0,0,1,0],[0, 0,0,-1],[1, 0, 0,0],[0,1, 0, 0]],
    [[0,0,0,1],[0, 0,1, 0],[0,-1, 0,0],[1,0, 0, 0]],
])

def multiply(q1, q2, renormalize=True):
    """
    Return the versor q resulting from multiplication of q1 with q2
    q represents a rotation first by q2 and then by q1 (order matters)
    q, q1, and q2 each have shape (..., 4), where ... are 0 or more leading batch dimensions
    if renormalize is True, q should be renormalized to unit length before it is returned
    """
    q3 = (_m * q1[...,None,:,None] * q2[...,None,None,:]).sum(axis=(-2, -1))
    if renormalize: q3 /= np.linalg.norm(q3, axis=-1, keepdims=True)
    return q3

if __name__ == "__main__":

    # you can edit this area for informal testing if you want
    axis = np.array([1., 1., 1.])
    angle = np.array(np.pi/3)
    quat = versor_from(axis, angle, normalize=True)
    mat = matrix_from(quat)
    vec1 = rotate(quat, axis)
    vec2 = mat @ axis
    print(axis)
    print(vec1)
    print(vec2)


