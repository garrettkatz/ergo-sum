import unittest as ut
import inspect
import numpy as np
import rotations as rt

class VersorFromTestCase(ut.TestCase):

    def test_invariant(self):
        axis = np.array([1., 1., 1.])
        angle = np.pi/3
        versor = rt.versor_from(axis.clone(), angle, normalize=True)
        result = rt.rotate(versor, axis)
        self.assertTrue(np.allclose(axis, result))

    def test_batching(self):
        axis = np.array([[1., 1., 1.]])
        angle = np.array([np.pi])
        batched = rt.versor_from(axis, angle, normalize=True)
        unbatched = rt.versor_from(axis[0], angle[0], normalize=True)
        self.assertTrue(np.allclose(batched[0], unbatched))

class RotateTestCase(ut.TestCase):

    def test_coordinate_axes(self):
        q1 = rt.versor_from(np.array([[0, 0, 1.]]), np.array([np.pi/2]))
        q2 = rt.versor_from(np.array([[1, 0, 0.]]), np.array([np.pi/2]))
        q3 = rt.multiply(q2, q1)
        vec = rt.rotate(q3, np.array([[1, 0, 0.]]))
        self.assertTrue(np.allclose(vec, np.array([[0, 0, 1.]])))

    def test_match_matrix(self):
        axis = np.randn(1,3)
        angle = np.randn(1)
        vec = np.randn(3)
        versor = rt.versor_from(axis, angle, normalize=True)
        mat = rt.matrix_from(versor)
        vec1 = rt.rotate(versor, vec)
        vec2 = mat @ vec
        self.assertTrue(np.allclose(vec1, vec2))


if __name__ == "__main__":

    test_suite = ut.TestLoader().loadTestsFromTestCase(ImportsTestCase)
    res = ut.TextTestRunner(verbosity=2).run(test_suite)

    if len(res.failures) > 0 or len(res.errors) > 0:

        print("score: 0 (no extra imports allowed)")

    else:

        num, errs, fails = 0, 0, 0
        test_cases = [VersorFromTestCase, RotateTestCase, AutoDiffTestCase]
    
        for test_case in test_cases:
            test_suite = ut.TestLoader().loadTestsFromTestCase(test_case)
            res = ut.TextTestRunner(verbosity=2).run(test_suite)
            num += res.testsRun
            errs += len(res.errors)
            fails += len(res.failures)
    
        print("score: %d of %d (%d errors, %d failures)" % (num - (errs+fails), num, errs, fails))


