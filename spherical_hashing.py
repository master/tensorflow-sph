"""Contains the definition and optimization procedure for Spherical hashing."""
import tensorflow.compat.v1 as tf

SAMPLE_SIZE = 5


def pairwise_distances(a, b):
    """Computes a matrix of pairwise distances between row vectors.

    Args:
      a: A float32 `Tensor` of shape [num_a, num_dims].
      b: A float32 `Tensor` of shape [num_b, num_dims].

    Returns:
      A float32 `Tensor` of shape [num_a, num_b].
    """
    anorm = tf.reshape(tf.reduce_sum(a * a, 1), shape=[-1, 1])
    bnorm = tf.reshape(tf.reduce_sum(b * b, 1), shape=[-1, 1])
    d = anorm - 2 * tf.matmul(a, tf.transpose(b)) + tf.transpose(bnorm)
    return tf.sqrt(d)


def spherical_hashing(
    inputs, num_bits, pack_bits=True, reuse=False, scope='SphericalHashing'
):
    """Spherical hashing from https://dl.acm.org/citation.cfm?id=2882197

    Heo, Jae-Pil, et al. "Spherical hashing." Computer Vision and Pattern
    Recognition (CVPR), 2012 IEEE Conference on. IEEE, 2012.

    Args:
      inputs: A float32 `Tensor` of input data.
      num_bits: Number of pivots (equivalently, number of hash bits).
      pack_bits: Whether to pack bits into int64s.
      reuse: Whether or not variables should be reused. To be able to reuse `scope`
        must be given.
      scope: Optional variable scope.

    Returns:
      A tuple of output bits and a dictionary of model variables.
    """
    with tf.variable_scope(scope, 'SphericalHashing', [inputs], reuse=reuse):
        inputs_dims = inputs.shape[1]
        pivots = tf.Variable(
            name='Pivots',
            shape=[num_bits, inputs_dims],
            initial_value=tf.zeros([num_bits, inputs_dims]),
            collections=[tf.GraphKeys.MODEL_VARIABLES],
        )
        radii = tf.Variable(
            name='Radii',
            shape=[num_bits, 1],
            initial_value=tf.zeros([num_bits, 1]),
            collections=[tf.GraphKeys.MODEL_VARIABLES],
        )
        distances = pairwise_distances(inputs, pivots)
        thresholds = distances - tf.transpose(radii)
        bits = tf.cast(thresholds <= 0, tf.uint8, name='Bits')
        if pack_bits:
            pow_2 = tf.pow(tf.to_int64(2), tf.range(0, 64, dtype=tf.int64))
            bits = tf.reshape(bits, [-1, 64])
            bits = tf.reduce_sum(tf.to_int64(bits) * pow_2, axis=1)
            bits = tf.reshape(bits, [-1, num_bits // 64])
        end_points = {'Pivots': pivots, 'Radii': radii}
        return bits, end_points


def train_spherical_hashing(
    step,
    inputs,
    num_bits,
    overlap_ratio=0.25,
    epsilon_avg=0.1,
    epsilon_stdev=0.15,
    seed=None,
    scope=None,
):
    """Training procedure for Spherical hashing.

    The optimization process is similar to the N-body simulation designed for
    simulating dynamic systems of particles (e.g., celestial objects
    interacting with each other under gravitational forces). See for details
    Greengard, Leslie. "The rapid evaluation of potential fields in particle
    systems." MIT press, 1988.

    Args:
      step: Training step variable.
      inputs: A float32 `Tensor` of training data.
      num_bits: Number of pivots (equivalently, number of hash bits).
      overlap_ratio: Constant to satisfy independence condition (see Eq.2 in
        the paper). Defaults to 1/4.
      epsilon_avg: Error tolerance for mean. Defaults to 10%.
      epsilon_stdev: Error tolerance for standard deviation. Defaults to 25%.
      seed: Optional random seed.
      scope: Optional variable scope.

    Returns:
      A tuple (`init_op`, `training_op`, `variables`).
        init_op: Training initialization operation.
        training_op: An operation that runs the optimizer training op and
          updates model variables.
        variables: A dictionary of training variables.
    """
    with tf.variable_scope(scope, 'SphericalHashing', [inputs]) as scope:
        _, end_points = spherical_hashing(inputs, num_bits, scope=scope)
        pivots, radii = end_points['Pivots'], end_points['Radii']
        init_op = tf.assign(pivots, random_pivots(inputs, num_bits, seed=seed))
        overlap, _, _, _ = compute_statistics(inputs, pivots, overlap_ratio)
        num_inputs = tf.shape(inputs)[0]
        forces = compute_forces(overlap, pivots, num_inputs, overlap_ratio)
        update_pivots = tf.assign_add(pivots, forces)
        _, new_radii, avg, stdev = compute_statistics(
            inputs, pivots, overlap_ratio
        )
        update_radii = tf.assign(radii, new_radii)
        update_step = tf.assign_add(step, 1)
        num_inputs = tf.to_float(num_inputs)
        avg_cond = avg <= epsilon_avg * overlap_ratio * num_inputs
        stdev_cond = stdev <= epsilon_stdev * overlap_ratio * num_inputs
        stopping_cond = tf.logical_and(
            avg_cond, stdev_cond, name='stopping_cond'
        )
        training_op = tf.group(
            update_pivots, update_radii, update_step, name='training_op'
        )
        variables = {
            'Average': avg,
            'Stdev': stdev,
            'StoppingCond': stopping_cond,
        }
        return init_op, training_op, variables


def random_pivots(
    inputs, num_pivots, sample_size=SAMPLE_SIZE, seed=None, scope=None
):
    """Pivots initialization function.

    Initial pivots are determined as centers of `sample_size` randomly sampled
    points.

    Args:
      inputs: A float32 `Tensor` of training data.
      num_pivots: Number of pivots to initialize.
      sample_size: Number of points to sample for each pivot.
      seed: Optional random seed.
      scope: Optional variable scope.

    Returns:
      1-D float32 `Tensor` of pivots.
    """
    with tf.variable_scope(scope, 'RandomPivots', [inputs]):
        num_inputs = tf.shape(inputs)[0]
        indices = tf.tile(tf.range(num_inputs), [num_pivots])
        indices = tf.random_shuffle(indices, seed=seed)
        indices = tf.gather(indices, tf.range(num_pivots * sample_size))
        samples = tf.gather(inputs, indices)
        samples = tf.reshape(samples, [sample_size, num_pivots, -1])
        pivots = tf.reduce_sum(samples, 0) / sample_size
        return pivots


def zero_diag(input):
    """Helper function that zeros matrix diagonal.

    Args:
      input: 2-D float32 `Tensor`.

    Returns:
      2-D float32 `Tensor` with diagonal zeroed.
    """
    return input - tf.diag(tf.diag_part(input))


def compute_statistics(inputs, pivots, overlap_ratio, scope=None):
    """Computes a matrix of hypersphere overlaps and associated statistics.

    Args:
      inputs: A float32 `Tensor` of shape [num_inputs, input_dims].
      pivots: A float32 `Tensor` of shape [num_pivots, input_dims].
      overlap_ratio: Constant to satisfy independence condition.
      scope: Optional variable scope.

    Returns:
      A tuple (`overlap`, `radii`, `avg`, `stdev`).
        overlap: 2-D `Tensor` of hypersphere overlaps.
        radii: 1-D `Tensor` of hypersphere radii.
        avg: Mean of the `overlap` `Tensor`.
        stdev: Standard deviation of the `overlap` `Tensor`.
    """
    with tf.variable_scope(scope, 'ComputeStatistics', [inputs, pivots]):
        dists = pairwise_distances(pivots, inputs)
        num_inputs = tf.shape(inputs)[0]
        median = (num_inputs + 1) // 2
        sorted_dists, _ = tf.nn.top_k(dists, median + 1)
        sorted_dists = tf.transpose(sorted_dists)
        radii = tf.gather(sorted_dists, [median])
        radii = tf.transpose(radii)
        inside_sphere = tf.cast(dists <= radii, tf.float32)
        overlap = tf.matmul(inside_sphere, tf.transpose(inside_sphere))
        num_pivots = tf.shape(pivots)[0]
        bits = tf.to_float(num_pivots * (num_pivots - 1))
        mean = tf.reduce_sum(zero_diag(overlap)) / bits
        stdev = tf.reduce_sum(zero_diag((overlap - mean) ** 2))
        stdev = tf.sqrt(stdev / bits)
        overlap_adj = tf.to_float(num_inputs) * overlap_ratio
        adj = tf.abs(overlap - overlap_adj)
        avg = tf.reduce_sum(zero_diag(adj)) / bits
        return overlap, radii, avg, stdev


def compute_forces(overlap, pivots, num_inputs, overlap_ratio, scope=None):
    """Computes accumulated forces based on pairwise hypersphere intersection.

    A (repulsive or attractive) force between two hyperspheres centered in
    corresponding pivots is computed based on the difference between their
    overlap versus desired overlap amount.

    An accumulated force is the average of all forces computed from all the
    other pivots.

    Args:
      overlap: A float32 `Tensor` of shape [num_pivots, num_pivots].
      pivots: A float32 `Tensor` of shape [num_pivots, input_dims].
      num_inputs: Number of training points.
      overlap_ratio: Constant to satisfy independence condition.
      scope: Optional variable scope.

    Returns:
      A float32 `Tensor` of shape [num_pivots, input_dims].
    """
    with tf.variable_scope(scope, 'ComputeForces', [overlap, pivots]):
        num_pivots = tf.shape(pivots)[0]
        indices = tf.reshape(tf.range(num_pivots), [-1, 1])
        indices = tf.reshape(tf.tile(indices, [1, num_pivots]), [-1])
        pivots_i = tf.tile(pivots, [num_pivots, 1])
        pivots_j = tf.gather(pivots, indices)
        overlap_adj = tf.to_float(num_inputs) * overlap_ratio
        forces = 0.5 * tf.multiply(
            tf.reshape(pivots_i - pivots_j, shape=[num_pivots, num_pivots, -1]),
            tf.expand_dims((overlap - overlap_adj) / overlap_adj, 2),
        )
        return tf.reduce_sum(forces / tf.to_float(num_pivots), 0)
