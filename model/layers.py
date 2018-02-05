import collections
import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.contrib.layers.python.layers import utils


class Block(collections.namedtuple('Block',
                                   ['scope', 'unit_fn', 'args'])):
    """A named tuple describing a mobilenet block"""


@slim.add_arg_scope
def bottleneck(
        inputs, depth, expansion, stride,
        outputs_collections=None, scope=None):
    """

    :param inputs:
    :param depth:
    :param expansion:
    :param stride:
    :param outputs_collections:
    :param scope:
    :return output:
    """
    with tf.variable_scope(scope, 'bottleneck', [inputs]) as sc:
        residual = inputs
        depth_in = utils.last_dimension(
            inputs.get_shape(), min_rank=4)

        output = slim.conv2d(
            inputs, expansion*depth_in, 1, 1, scope='conv1x1_1')

        output = slim.batch_norm(
            output, scope='conv1x1_1_bn')

        output = tf.nn.relu6(output)

        output = slim.separable_conv2d(
            output, depth, 3, 1, stride, scope='separable_conv3x3')

        output = slim.batch_norm(
            output, scope='separable_conv3x3_bn')

        output = tf.nn.relu6(output)

        output = slim.conv2d(
            output, depth, 1, 1, scope='conv1x1_2')

        output = slim.batch_norm(output, scope='conv1x1_2_bn')

        if stride == 1:
            output += residual

        return utils.collect_named_outputs(outputs_collections,
                                           sc.name, output)


@slim.add_arg_scope
def stack_blocks(
        net, blocks, outputs_collections=None):
    """

    :param net:
    :param blocks:
    :param outputs_collections:
    :return net:
    """
    for block in blocks:
        with tf.variable_scope(block.scope, 'block', [net]) as sc:
            for i, unit in enumerate(block.args):
                with tf.variable_scope('unit_%d' % (i+1), values=[net]):
                    unit_depth, unit_expansion, unit_stride = unit
                    net = block.unit_fn(
                        net, depth=unit_depth, expansion=unit_expansion,
                        stride=unit_stride)

            net = utils.collect_named_outputs(outputs_collections, sc.name, net)
    return net


def mobile_arg_scope(
        is_training=True, weight_decay=1e-4, bn_decay=0.997, bn_epsilon=1e-5):
    """

    :param is_training:
    :param weight_decay:
    :param bn_decay:
    :param bn_epsilon:
    :return arg_sc:
    """
    batch_norm_params = {
        'is_training': is_training,
        'decay': bn_decay,
        'epsilon': bn_epsilon,
        'scale': True,
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
        'fused': True,
    }

    with slim.arg_scope(
        [slim.conv2d, slim.separable_conv2d],
        weights_regularizer=slim.l2_regularizer(weight_decay),
        weights_initializer=slim.variance_scaling_initializer(),
        activation_fn=None,
        normallizer_fn=None
    ):
        with slim.arg_scope(
            [slim.batch_norm], **batch_norm_params
        ) as arg_sc:
            return arg_sc


def mobile(
        inputs, blocks, root=None,
        num_class=0, training=True,
        reuse=None, scope=None):
    """

    :param inputs:
    :param blocks:
    :param reuse:
    :param scope:
    :param root:
    :param num_class:
    :param training:
    :return net:
    :return end_points:
    """
    with tf.variable_scope(
            scope, 'mobilenet', [inputs], reuse=reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope(
                [slim.conv2d, bottleneck, stack_blocks, slim.avg_pool2d,
                 slim.dropout, slim.fully_connected],
                outputs_collections=end_points_collection
        ):
            net = inputs
            net = slim.conv2d(
                net, 32, 3, 2, scope='conv1x1_1')
            net = slim.batch_norm(
                net, scope='conv1x1_1_bn')

            net = stack_blocks(net, blocks)

            if root:
                net = slim.conv2d(
                    net, 1280, 1, 1, scope='conv1_1x1_2')

                net = slim.batch_norm(
                    net, scope='conv1_1x1_2_bn')

                net = slim.avg_pool2d(
                    net, 7, stride=1, padding='SAME', scope='pool')
                
                net = slim.dropout(
                    net, 0.2, is_training=training, scope='dropout')

            if num_class > 0:
                net = slim.fully_connected(net, num_class, scope='fc')

            end_points = utils.convert_collection_to_dict(end_points_collection)

            return net, end_points
