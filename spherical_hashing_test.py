"""Tests for Spherical hashing."""

import tensorflow.compat.v1 as tf

tf.disable_eager_execution()

import numpy as np

from . import spherical_hashing


class SphericalHashingTest(tf.test.TestCase):
    def test_pairwise_distances(self):
        data = np.array([1, 2, 3, 4, 5, 6, 7.1, 2])
        # fmt: off
        dist_ref = np.array([[0.00000, 2.82843, 5.65685, 6.10000],
                             [2.82843, 0.00000, 2.82843, 4.56180],
                             [5.65685, 2.82843, 0.00000, 4.51774],
                             [6.10000, 4.56180, 4.51774, 0.00000]])
        # fmt: on
        with self.test_session() as sess:
            data = tf.constant(data, shape=[4, 2], dtype=tf.float32)
            dist = spherical_hashing.pairwise_distances(data, data)
            dist_np = sess.run(dist)
        self.assertAllClose(dist_np, dist_ref)

    def test_zero_diag(self):
        # fmt: off
        data = np.array([[1, 2, 3],
                         [3, 4, 5],
                         [6, 7, 8]])
        diag_ref = np.array([[0, 2, 3],
                             [3, 0, 5],
                             [6, 7, 0]])
        # fmt: on
        with self.test_session() as sess:
            data = tf.constant(data, shape=[3, 3], dtype=tf.float32)
            diag = spherical_hashing.zero_diag(data)
            diag_np = sess.run(diag)
        self.assertAllClose(diag_np, diag_ref)

    def test_compute_statistics(self):
        inputs = np.array([1, 2, 3, 4, 5, 6, 7.1, 2])
        pivots = np.array([3, 1, 3, 4.1, 5, 5])
        radii_ref = np.array([[3.0000], [2.758623], [2.236068]])
        with self.test_session() as sess:
            inputs = tf.constant(inputs, shape=[4, 2], dtype=tf.float32)
            pivots = tf.constant(pivots, shape=[3, 2], dtype=tf.float32)
            _, radii, avg, stdev = spherical_hashing.compute_statistics(
                inputs, pivots, overlap_ratio=0.25
            )
            np_radii, np_avg, np_stdev = sess.run([radii, avg, stdev])
        self.assertAllClose(np_radii, radii_ref)
        self.assertAllClose(np_avg, 0.3333333)
        self.assertAllClose(np_stdev, 0.47140449)

    def test_compute_forces(self):
        # fmt: off
        pivots = np.array([3, 1, 3, 4.1, 5, 100])
        overlap = np.array([[2., 1., 0.],
                            [1., 2., 1.],
                            [0., 1., 2.]])
        forces_ref = np.array([[ 0.33333334, 16.98627472],
                               [ 0.3137255,  14.55686283],
                               [-0.64705884,-31.5431366]])
        # fmt: on
        with self.test_session() as sess:
            pivots = tf.constant(pivots, shape=[3, 2], dtype=tf.float32)
            overlap = tf.constant(overlap, shape=[3, 3], dtype=tf.float32)
            forces = spherical_hashing.compute_forces(
                overlap, pivots, num_inputs=1, overlap_ratio=17
            )
            np_forces = sess.run(forces)
        self.assertAllClose(np_forces, forces_ref)

    def test_train_spherical_hashing(self):
        seed = 1
        train = tf.random_uniform(shape=[20000, 128], seed=seed)
        step = tf.train.get_or_create_global_step()
        train_ops = spherical_hashing.train_spherical_hashing(
            step, train, 64, seed=seed
        )
        init_op, training_op, variables = train_ops
        bits, end_points = spherical_hashing.spherical_hashing(
            tf.random_uniform(shape=[1, 128], seed=seed),
            64,
            pack_bits=False,
            reuse=True,
        )
        packed_bits, _ = spherical_hashing.spherical_hashing(
            tf.random_uniform(shape=[1, 128], seed=seed),
            64,
            pack_bits=True,
            reuse=True,
        )
        pivots_ref = np.array(
            [
                0.688757,
                0.708312,
                0.160226,
                1.234819,
                0.534309,
                0.459139,
                0.403541,
                0.830583,
                0.568704,
                -0.039965,
                0.208738,
                0.761988,
                0.400137,
                0.787016,
                0.903489,
                1.122519,
            ],
            dtype=np.float32,
        )
        radii_ref = np.array([4.715613], dtype=np.float32)
        bits_ref = np.array(
            [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0], dtype=np.uint8
        )
        packed_bits_ref = 3404227049824520
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(init_op)
            while True:
                try:
                    _, step_np = sess.run([training_op, step])
                except tf.errors.InvalidArgumentError:
                    break
                if step_np > 20:
                    break
            np_pivots = sess.run(end_points['Pivots'])
            np_radii = sess.run(end_points['Radii'])
            np_bits = sess.run(bits)
            np_packed_bits = sess.run(packed_bits)
        self.assertEqual(step_np, 9)
        self.assertAllClose(np_pivots[0][0:16], pivots_ref)
        self.assertAllClose(np_radii[0], radii_ref)
        self.assertAllClose(np_bits[0][0:16], bits_ref)
        self.assertEqual(np_packed_bits[0], packed_bits_ref)
