import unittest as ut
import inspect
import numpy as np
import scipy.optimize as so
import torch as tr
tr.set_default_dtype(tr.float64) # makes round-off errors much smaller
import rotations as rt
import forward_kinematics as fk
import inverse_kinematics as ik
import ergo_jr

header = """# Implement the stubs where requested in the comments below
# No imports allowed other than these, do not change these lines
import numpy as np
import scipy.optimize as so
import matplotlib.pyplot as pt
import torch as tr
tr.set_default_dtype(tr.float64) # makes round-off errors much smaller
import forward_kinematics as fk
import ergo_jr
"""

class ImportsTestCase(ut.TestCase):

    def test_imports(self):

        trg = header.split("\n")
        src, _ = inspect.getsourcelines(ik)
        for t in range(len(trg)):
            self.assertEqual(src[t].strip(), trg[t])
        self.assertFalse("import" in "".join(src[len(trg):]))

# helper for comparing tensors
def allclose(a, b):
    return (a.round(decimals=4) == b.round(decimals=4)).all()

class LocationConstraintTestCase(ut.TestCase):

    def test_zero(self):
        angles = tr.zeros(6)
        targets, _ = fk.get_frames(ergo_jr.joint_info, angles)
        for i, target in enumerate(targets):
            sq_dist = ik.location_constraint_fun(ergo_jr.joint_info, i, target, angles)
            jacobian = ik.location_constraint_jac(ergo_jr.joint_info, i, target, angles)
            self.assertTrue(allclose(sq_dist, tr.zeros(1)))
            self.assertTrue(allclose(jacobian, tr.zeros(1,6)))

    def test_nonzero(self):
        target = tr.tensor([.0, -.07, 0])
        angles = tr.zeros(6)
        sq_dist = ik.location_constraint_fun(ergo_jr.joint_info, 5, target, angles)
        jacobian = ik.location_constraint_jac(ergo_jr.joint_info, 5, target, angles)
        self.assertTrue(allclose(sq_dist, tr.tensor([.0351])))
        self.assertTrue(allclose(jacobian, tr.tensor([[ 0.0022, -0.0335, -0.0446,  0.0022, -0.0389,  0.0000]])))

        target = tr.tensor([.1, .2, .3])
        angles = tr.full((6,), .2)
        sq_dist = ik.location_constraint_fun(ergo_jr.joint_info, 4, target, angles)
        jacobian = ik.location_constraint_jac(ergo_jr.joint_info, 4, target, angles)
        self.assertTrue(allclose(sq_dist, tr.tensor([0.1057])))
        self.assertTrue(allclose(jacobian, tr.tensor([[-0.0231,  0.0616,  0.0318, -0.0177,  0.0000,  0.0000]])))

        self.assertTrue(tuple(sq_dist.shape) == (1,))
        self.assertTrue(tuple(jacobian.shape) == (1,6))


class InverseKinematicsTestCase(ut.TestCase):

    def test_zeros(self):

        targets, _ = fk.get_frames(ergo_jr.joint_info, tr.zeros(6))

        soln = so.minimize(
            ik.angle_norm_obj_and_grad,
            x0 = np.full((6,), .01),
            jac = True,
            constraints = [
                ik.location_constraint(ergo_jr.joint_info, i, targets[i])
                for i in (5, 7)
            ],
        )

        self.assertTrue(soln.message == "Optimization terminated successfully")    
        angles = tr.tensor(soln.x).round(decimals=3)
        self.assertTrue(allclose(angles, tr.zeros(6)))

    # TEMPLATE-START
    # this test is not portable, optimization does not converge on surface
    # def test_finger_targets(self):

    #     all_targets = tr.tensor([
    #         [[.0, -.07, 0],
    #          [-.02, -.07, 0]],
        
    #         [[-.03, -.07, 0],
    #          [-.05, -.07, 0]],
        
    #         [[-.03, -.07, 0],
    #          [-.05, -.05, 0]],
    #     ])

    #     for (target_5, target_7) in all_targets:

    #         soln = so.minimize(
    #             ik.angle_norm_obj_and_grad,
    #             x0 = np.zeros(6),
    #             jac = True,
    #             bounds = [(-np.pi, np.pi)] * 6,
    #             constraints = [
    #                 ik.location_constraint(ergo_jr.joint_info, 5, target_5),
    #                 ik.location_constraint(ergo_jr.joint_info, 7, target_7),
    #             ],
    #             options={'maxiter': 200},
    #         )

    #         self.assertTrue(soln.message == "Optimization terminated successfully")        
    
    #         angles = tr.tensor(soln.x)
    #         locations, orientations = fk.get_frames(ergo_jr.joint_info, angles)
    #         self.assertTrue(allclose(locations[5], target_5))
    #         self.assertTrue(allclose(locations[7], target_7))
    # TEMPLATE-END

if __name__ == "__main__":

    test_suite = ut.TestLoader().loadTestsFromTestCase(ImportsTestCase)
    res = ut.TextTestRunner(verbosity=2).run(test_suite)

    if len(res.failures) > 0 or len(res.errors) > 0:

        print("score: 0 (no extra imports allowed)")

    else:

        num, errs, fails = 0, 0, 0
        test_cases = [LocationConstraintTestCase, InverseKinematicsTestCase]
    
        for test_case in test_cases:
            test_suite = ut.TestLoader().loadTestsFromTestCase(test_case)
            res = ut.TextTestRunner(verbosity=2).run(test_suite)
            num += res.testsRun
            errs += len(res.errors)
            fails += len(res.failures)
    
        print("score: %d of %d (%d errors, %d failures)" % (num - (errs+fails), num, errs, fails))






