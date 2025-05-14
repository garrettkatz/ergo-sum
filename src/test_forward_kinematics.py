import unittest as ut
import inspect
import torch as tr
import rotations as rt
import forward_kinematics as fk
import ergo_jr

header = """# Implement the stubs where requested in the comments below
# No imports allowed other than these, do not change these lines
import matplotlib.pyplot as pt
import torch as tr
tr.set_default_dtype(tr.float64) # makes round-off errors much smaller
import rotations as rt
import ergo_jr
"""

class ImportsTestCase(ut.TestCase):

    def test_imports(self):

        trg = header.split("\n")
        src, _ = inspect.getsourcelines(fk)
        for t in range(len(trg)):
            self.assertEqual(src[t].strip(), trg[t])
        self.assertFalse("import" in "".join(src[len(trg):]))

# helper for comparing tensors
def allclose(a, b):
    return (a.round(decimals=4) == b.round(decimals=4)).all()

class GetFramesTestCase(ut.TestCase):

    def _test_angles(self, angles, expected_locations, expected_orientations):
        locations, orientations = fk.get_frames(ergo_jr.joint_info, angles)
        self.assertTrue(allclose(locations, expected_locations))
        self.assertTrue(allclose(orientations, expected_orientations))

    def test_zeros(self):
        self._test_angles(
            angles = tr.zeros(6),
            expected_locations = tr.tensor([
                [ 0.,      0.,      0.0328],
                [ 0.,      0.,      0.0568],
                [-0.,      0.,      0.1108],
                [-0.,      0.,      0.1558],
                [-0.,     -0.048,   0.1558],
                [ 0.0155, -0.173,   0.1558],
                [-0.,     -0.106,   0.1558],
                [-0.0155, -0.1735,  0.1558]]),
           expected_orientations = tr.tensor([
                [ 1.,       0.,       0.,       0.],
                [ 0.70711,  0.,      -0.70711,  0.],
                [ 0.70711,  0.,      -0.70711,  0.],
                [ 0.,       0.,       1.,       0.],
                [ 0.70711,  0.,       0.70711,  0.],
                [ 0.70711,  0.,       0.70711,  0.],
                [ 1.,       0.,      -0.,       0.],
                [ 1.,       0.,      -0.,       0.]]))

    def test_neg(self):
        self._test_angles(
            angles = tr.full((6,), -.2),
            expected_locations = tr.tensor([
                [ 0.,      0.,      0.0328],
                [ 0.,      0.,      0.0568],
                [ 0.0021,  0.0105,  0.1097],
                [ 0.0056,  0.0277,  0.1512],
                [-0.0123, -0.0129,  0.1695],
                [-0.0419, -0.1128,  0.2403],
                [-0.0327, -0.0565,  0.2018],
                [-0.0587, -0.108,   0.2402]]),
            expected_orientations = tr.tensor([
                [ 1.,      0.,      0.,      0.    ],
                [ 0.7036, -0.0706, -0.7036, -0.0706],
                [ 0.7071, -0.1405, -0.693,   0.    ],
                [-0.0198,  0.0978,  0.9752, -0.1977],
                [ 0.6792, -0.0028,  0.6792, -0.2782],
                [ 0.648,  -0.0706,  0.6755, -0.3446],
                [ 0.9359, -0.2936,  0.0194, -0.1937],
                [ 0.9506, -0.2902,  0.0486, -0.0993]]))

    def test_pos(self):
        self._test_angles(
            angles = tr.full((6,), +.2),
            expected_locations = tr.tensor([
                [ 0.,      0.,      0.0328],
                [ 0.,      0.,      0.0568],
                [ 0.0021, -0.0105,  0.1097],
                [ 0.0056, -0.0277,  0.1512],
                [ 0.0236, -0.0683,  0.1329],
                [ 0.0818, -0.1565,  0.0644],
                [ 0.0439, -0.1119,  0.1005],
                [ 0.0397, -0.1701,  0.0632]]),
            expected_orientations = tr.tensor([
                [ 1.,      0.,      0.,      0.    ],
                [ 0.7036,  0.0706, -0.7036,  0.0706],
                [ 0.7071,  0.1405, -0.693,   0.    ],
                [-0.0198, -0.0978,  0.9752,  0.1977],
                [ 0.6792,  0.0028,  0.6792,  0.2782],
                [ 0.648,   0.0706,  0.6755,  0.3446],
                [ 0.9359,  0.2936,  0.0194,  0.1937],
                [ 0.9506,  0.2902,  0.0486,  0.0993]]))

    def test_batched(self):
        angle_batch = tr.tensor([[-.2, 0, +.2]]).t() * tr.ones((1,6))

        # get expected results for unbatched calls
        expected_locations = []
        expected_orientations = []
        for angles in angle_batch:
            locations, orientations = fk.get_frames(ergo_jr.joint_info, angles)
            expected_locations.append(locations)
            expected_orientations.append(orientations)

        # compare with a batched call
        expected_locations = tr.stack(expected_locations)
        expected_orientations = tr.stack(expected_orientations)
        self._test_angles(angle_batch, expected_locations, expected_orientations)        

class GetJacobianTestCase(ut.TestCase):

    def test_unbatched(self):
        angle_batch = tr.tensor([[-.2, 0, +.2]]).t() * tr.ones((1,6))
        for angles in angle_batch:
            locations, orientations = fk.get_frames(ergo_jr.joint_info, angles)
            jacobian = fk.get_jacobian(ergo_jr.joint_info, locations, orientations)
            jacobian_t = tr.autograd.functional.jacobian(lambda a: fk.get_frames(ergo_jr.joint_info, a)[0], angles)
            self.assertTrue(allclose(jacobian, jacobian_t))

    def test_batched(self):
        angle_batch = tr.tensor([[-.5, -.1, 0., +.1, +.5]]).t() * tr.ones((1,6))
        locations, orientations = fk.get_frames(ergo_jr.joint_info, angle_batch)
        jacobian = fk.get_jacobian(ergo_jr.joint_info, locations, orientations)

        jacobian_t = []
        for angles in angle_batch:
            jac = tr.autograd.functional.jacobian(lambda a: fk.get_frames(ergo_jr.joint_info, a)[0], angles)
            jacobian_t.append(jac)
        jacobian_t = tr.stack(jacobian_t)

        self.assertTrue(allclose(jacobian, jacobian_t))


class AutoDiffTestCase(ut.TestCase):

    def test_gradient_flow(self):
        angle_batch = tr.randn(4, 6, requires_grad=True)
        locations, orientations = fk.get_frames(ergo_jr.joint_info, angle_batch)
        locations.sum().backward()
        self.assertTrue(angle_batch.grad is not None)

        angle_batch = tr.randn(4, 6, requires_grad=True)
        locations, orientations = fk.get_frames(ergo_jr.joint_info, angle_batch)
        jacobian = fk.get_jacobian(ergo_jr.joint_info, locations, orientations)
        jacobian.sum().backward()
        self.assertTrue(angle_batch.grad is not None)

if __name__ == "__main__":

    test_suite = ut.TestLoader().loadTestsFromTestCase(ImportsTestCase)
    res = ut.TextTestRunner(verbosity=2).run(test_suite)

    if len(res.failures) > 0 or len(res.errors) > 0:

        print("score: 0 (no extra imports allowed)")

    else:

        num, errs, fails = 0, 0, 0
        test_cases = [GetFramesTestCase, GetJacobianTestCase, AutoDiffTestCase]
    
        for test_case in test_cases:
            test_suite = ut.TestLoader().loadTestsFromTestCase(test_case)
            res = ut.TextTestRunner(verbosity=2).run(test_suite)
            num += res.testsRun
            errs += len(res.errors)
            fails += len(res.failures)
    
        print("score: %d of %d (%d errors, %d failures)" % (num - (errs+fails), num, errs, fails))




