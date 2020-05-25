import tensorflow as tf
import numpy as np

from tensorflow.contrib import slim


tf.app.flags.DEFINE_integer('text_scale', 512, '')


from nets import resnet_v1

FLAGS = tf.app.flags.FLAGS


# def unpool(inputs):
#     return tf.image.resize_bilinear(inputs, size=[tf.shape(inputs)[1]*2,  tf.shape(inputs)[2]*2])
def unpool(inputs,scale):
    return tf.image.resize_bilinear(inputs, size=[tf.shape(inputs)[1]*scale,  tf.shape(inputs)[2]*scale])



def ResidualConvUnit(inputs,features=256,kernel_size=3):
    net=tf.nn.relu(inputs)
    net=slim.conv2d(net, features, kernel_size)
    net=tf.nn.relu(net)
    net=slim.conv2d(net,features,kernel_size)
    net=tf.add(net,inputs)
    return net

def ChainedResidualPooling(inputs,features=256):
    net_relu=tf.nn.relu(inputs)
    net=slim.max_pool2d(net_relu, [5, 5],stride=1,padding='SAME')
    net=slim.conv2d(net,features,3)
    net_sum_1=tf.add(net,net_relu)

    net = slim.max_pool2d(net_relu, [5, 5], stride=1, padding='SAME')
    net = slim.conv2d(net, features, 3)
    net_sum_2=tf.add(net,net_sum_1)

    return net_sum_2


def MultiResolutionFusion(high_inputs=None,low_inputs=None,features=256):

    if high_inputs is None:#refineNet block 4
        rcu_low_1 = low_inputs[0]
        rcu_low_2 = low_inputs[1]

        rcu_low_1 = slim.conv2d(rcu_low_1, features, 3)
        rcu_low_2 = slim.conv2d(rcu_low_2, features, 3)

        return tf.add(rcu_low_1,rcu_low_2)

    else:
        rcu_low_1 = low_inputs[0]
        rcu_low_2 = low_inputs[1]

        rcu_low_1 = slim.conv2d(rcu_low_1, features, 3)
        rcu_low_2 = slim.conv2d(rcu_low_2, features, 3)

        rcu_low = tf.add(rcu_low_1,rcu_low_2)

        rcu_high_1 = high_inputs[0]
        rcu_high_2 = high_inputs[1]

        rcu_high_1 = unpool(slim.conv2d(rcu_high_1, features, 3),2)
        rcu_high_2 = unpool(slim.conv2d(rcu_high_2, features, 3),2)

        rcu_high = tf.add(rcu_high_1,rcu_high_2)

        return tf.add(rcu_low, rcu_high)


def RefineBlock(high_inputs=None,low_inputs=None):

    if high_inputs is None: # block 4
        rcu_low_1= ResidualConvUnit(low_inputs, features=256)
        rcu_low_2 = ResidualConvUnit(low_inputs, features=256)
        rcu_low = [rcu_low_1, rcu_low_2]

        fuse = MultiResolutionFusion(high_inputs=None, low_inputs=rcu_low, features=256)
        fuse_pooling = ChainedResidualPooling(fuse, features=256)
        output = ResidualConvUnit(fuse_pooling, features=256)
        return output
    else:
        rcu_low_1 = ResidualConvUnit(low_inputs, features=256)
        rcu_low_2 = ResidualConvUnit(low_inputs, features=256)
        rcu_low = [rcu_low_1, rcu_low_2]

        rcu_high_1 = ResidualConvUnit(high_inputs, features=256)
        rcu_high_2 = ResidualConvUnit(high_inputs, features=256)
        rcu_high = [rcu_high_1, rcu_high_2]

        fuse = MultiResolutionFusion(rcu_high, rcu_low,features=256)
        fuse_pooling = ChainedResidualPooling(fuse, features=256)
        output = ResidualConvUnit(fuse_pooling, features=256)
        return output





def mean_image_subtraction(images, means=[123.68, 116.78, 103.94]):
    '''
    image normalization
    :param images:
    :param means:
    :return:
    '''
    images=tf.to_float(images)
    num_channels = images.get_shape().as_list()[-1]
    if len(means) != num_channels:
      raise ValueError('len(means) must match the number of channels')
    channels = tf.split(axis=3, num_or_size_splits=num_channels, value=images)
    for i in range(num_channels):
        channels[i] -= means[i]
    return tf.concat(axis=3, values=channels)

def model(images, weight_decay=1e-5, is_training=True):
    images = mean_image_subtraction(images)

    with slim.arg_scope(resnet_v1.resnet_arg_scope(weight_decay=weight_decay)):
        logits, end_points = resnet_v1.resnet_v1_50(images, is_training=is_training, scope='resnet_v1_50')

    with tf.variable_scope('feature_fusion', values=[end_points.values]):
        batch_norm_params = {'decay': 0.997,'epsilon': 1e-5,'scale': True,'is_training': is_training}
        with slim.arg_scope([slim.conv2d],
                            activation_fn=tf.nn.relu,
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            weights_regularizer=slim.l2_regularizer(weight_decay)):

            f = [end_points['pool5'], end_points['pool4'],
                 end_points['pool3'], end_points['pool2']]
            for i in range(4):
                print('Shape of f_{} {}'.format(i, f[i].shape))

            g = [None, None, None, None]
            h = [None, None, None, None]

            for i in range(4):
                h[i]=slim.conv2d(f[i], 256, 1)
            for i in range(4):
                print('Shape of h_{} {}'.format(i, h[i].shape))

            g[0]=RefineBlock(high_inputs=None,low_inputs=h[0])
            print('Shape of g_{} {}'.format(0, g[0].shape))
            g[1]=RefineBlock(g[0],h[1])
            print('Shape of g_{} {}'.format(1, g[1].shape))
            g[2]=RefineBlock(g[1],h[2])
            print('Shape of g_{} {}'.format(2, g[2].shape))
            g[3]=RefineBlock(g[2],h[3])
            g[3] = slim.conv2d(g[3], 128, 3)
            g[3] = slim.conv2d(g[3], 64, 3)
            g[3] = slim.conv2d(g[3], 32, 3)
            print('Shape of g_{} {}'.format(3, g[3].shape))

            #g[3]=unpool(g[3],scale=4)
            #g[3] = horizontal_vertical_lstm_together(g[3], 128, scope_n="layer1")
            F_score = slim.conv2d(g[3], 1, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None)
            geo_map = slim.conv2d(g[3], 4, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None) * FLAGS.text_scale
            angle_map = (slim.conv2d(g[3], 1, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None) - 0.5) * np.pi/2 # angle is between [-45, 45]
            F_geometry = tf.concat([geo_map, angle_map], axis=-1)
            #F_score = slim.conv2d(g[3], 21, 1, activation_fn=tf.nn.relu, normalizer_fn=None)

    return F_score, F_geometry
# def model(images, weight_decay=1e-5, is_training=True):
#     '''
#     define the model, we use slim's implemention of resnet
#     '''
#     images = mean_image_subtraction(images)

#     with slim.arg_scope(resnet_v1.resnet_arg_scope(weight_decay=weight_decay)):
#         logits, end_points = resnet_v1.resnet_v1_50(images, is_training=is_training, scope='resnet_v1_50')

#     with tf.variable_scope('feature_fusion', values=[end_points.values]):
#         batch_norm_params = {
#         'decay': 0.997,
#         'epsilon': 1e-5,
#         'scale': True,
#         'is_training': is_training
#         }
#         with slim.arg_scope([slim.conv2d],
#                             activation_fn=tf.nn.relu,
#                             normalizer_fn=slim.batch_norm,
#                             normalizer_params=batch_norm_params,
#                             weights_regularizer=slim.l2_regularizer(weight_decay)):
#             f = [end_points['pool5'], end_points['pool4'],
#                  end_points['pool3'], end_points['pool2']]
#             for i in range(4):
#                 print('Shape of f_{} {}'.format(i, f[i].shape))
#             g = [None, None, None, None]
#             h = [None, None, None, None]
#             num_outputs = [None, 128, 64, 32]
#             for i in range(4):
#                 if i == 0:
#                     h[i] = f[i]
#                 else:
#                     c1_1 = slim.conv2d(tf.concat([g[i-1], f[i]], axis=-1), num_outputs[i], 1)
#                     h[i] = slim.conv2d(c1_1, num_outputs[i], 3)
#                 if i <= 2:
#                     g[i] = unpool(h[i])
#                 else:
#                     g[i] = slim.conv2d(h[i], num_outputs[i], 3)
#                 print('Shape of h_{} {}, g_{} {}'.format(i, h[i].shape, i, g[i].shape))

#             #here we use a slightly different way for regression part,
#             #we first use a sigmoid to limit the regression range, and also
#             #this is do with the angle map
#             #g[3] = horizontal_vertical_lstm_together(g[3], 128, scope_n="layer1")
#             F_score = slim.conv2d(g[3], 1, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None)
#             #4 channel of axis aligned bbox and 1 channel rotation angle
#             geo_map = slim.conv2d(g[3], 4, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None) * FLAGS.text_scale
#             angle_map = (slim.conv2d(g[3], 1, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None) - 0.5) * np.pi/2 # angle is between [-45, 45]
#             F_geometry = tf.concat([geo_map, angle_map], axis=-1)
            

#     return F_score, F_geometry



# def horizontal_vertical_lstm_together(input_data, rnn_size, scope_n="layer1"):
#     with tf.variable_scope("MultiDimensionalLSTMCell-horizontal-" + scope_n):
#         #input is (b, h, w, c)
#         #horizontal
#         _, _, _, c_h = input_data.get_shape().as_list()
#         shape_h=tf.shape(input_data)
#         b_h, h_h, w_h= shape_h[0],shape_h[1],shape_h[2]
#         #transpose = swap h and w.
#         new_input_data_h = tf.reshape(input_data, (b_h*h_h, w_h, c_h))  # horizontal.
#         #Forward
#         lstm_fw_cell = tf.contrib.rnn.LSTMCell(rnn_size//4)
#         lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(lstm_fw_cell, output_keep_prob=0.5)
#         #Backward
#         lstm_bw_cell = tf.contrib.rnn.LSTMCell(rnn_size//4)
#         lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(lstm_bw_cell, output_keep_prob=0.5)
 
 
#         rnn_out_h, _ = tf.nn.bidirectional_dynamic_rnn(
#                 lstm_fw_cell, 
#                 lstm_bw_cell, 
#                 inputs=new_input_data_h,
#                 dtype=tf.float32, 
#                 time_major=False)
#         rnn_out_h=tf.concat(rnn_out_h, 2)
#         rnn_out_h = tf.reshape(rnn_out_h, (-1, h_h, w_h, rnn_size//2))
#         #vertical
#     with tf.variable_scope("MultiDimensionalLSTMCell-vertical-" + scope_n):
#         new_input_data_v=tf.transpose(input_data,(0,2,1,3))
#         _, _, _, c_v = new_input_data_v.get_shape().as_list()
#         shape_v=tf.shape(new_input_data_v)
#         b_v, h_v, w_v = shape_v[0],shape_v[1],shape_v[2]
#         new_input_data_v = tf.reshape(new_input_data_v, (b_v*h_v, w_v, c_v))
#         #Forward
#         lstm_fw_cell = tf.contrib.rnn.LSTMCell(rnn_size//4)
#         lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(lstm_fw_cell, output_keep_prob=0.5)
#         #Backward
#         lstm_bw_cell = tf.contrib.rnn.LSTMCell(rnn_size//4)
#         lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(lstm_bw_cell, output_keep_prob=0.5)
 
 
#         rnn_out_v, _ = tf.nn.bidirectional_dynamic_rnn(
#                 lstm_fw_cell, 
#                 lstm_bw_cell, 
#                 inputs=new_input_data_v,
#                 dtype=tf.float32, 
#                 time_major=False)
#         rnn_out_v=tf.concat(rnn_out_v, 2)
        
#         rnn_out_v = tf.reshape(rnn_out_v, (-1, h_v, w_v, rnn_size//2))
#         rnn_out_v=tf.transpose(rnn_out_v,(0,2,1,3))
#         rnn_out=tf.concat([rnn_out_h,rnn_out_v],axis=3)
#         rnn_out=tf.add(rnn_out_h,rnn_out_v)
#         return rnn_out



# def focal_loss(y_true_cls,y_pred_cls,training_mask):
#     gamma = 0.75
#     alpha = 0.25
#     pt_1 = tf.where(tf.equal((y_true_cls * training_mask),1),(y_pred_cls * training_mask),tf.ones_like((y_pred_cls * training_mask)))
#     pt_0 = tf.where(tf.equal((y_true_cls* training_mask), 0), (y_pred_cls * training_mask), tf.zeros_like((y_pred_cls * training_mask)))
#     pt_1 = tf.clip_by_value(pt_1, 1e-3, .999)
#     pt_0 = tf.clip_by_value(pt_0, 1e-3, .999)
#     loss = -tf.reduce_sum(alpha * tf.pow(1. - pt_1, gamma) * tf.log(pt_1))-tf.reduce_sum((1-alpha) * tf.pow( pt_0, gamma) * tf.log(1. - pt_0))

#     return loss

 




def dice_coefficient(y_true_cls, y_pred_cls,
                     training_mask):
    '''
    dice loss
    :param y_true_cls:
    :param y_pred_cls:
    :param training_mask:
    :return:
    '''
    eps = 1e-5
    intersection = tf.reduce_sum(y_true_cls * y_pred_cls * training_mask)
    union = tf.reduce_sum(y_true_cls * training_mask) + tf.reduce_sum(y_pred_cls * training_mask) + eps
    loss = 1. - (2 * intersection / union)
    tf.summary.scalar('classification_dice_loss', loss)
    return loss
# def cross_loss(annotation_batch,upsampled_logits_batch,class_labels):
#     valid_labels_batch_tensor, valid_logits_batch_tensor = get_valid_logits_and_labels(
#         annotation_batch_tensor=annotation_batch,
#         logits_batch_tensor=upsampled_logits_batch,
#         class_labels=class_labels)

#     cross_entropies = tf.nn.softmax_cross_entropy_with_logits(logits=valid_logits_batch_tensor,
#                                                               labels=valid_labels_batch_tensor)

#     cross_entropy_sum = tf.reduce_mean(cross_entropies)
#     tf.summary.scalar('cross_entropy_loss', cross_entropy_sum)

#     return cross_entropy_sum



# def distance_balanced_sigmoid_crossentrop(y_true_cls, y_pred_cls,       #二分类平衡交叉熵损失
#                      training_mask):
#     eps = 1e-5
#     labels=y_true_cls*training_mask
#     logits=tf.nn.sigmoid(y_pred_cls*training_mask)
#     #min_distance=tf.abs(tf.minimum(tf.minimum(y_true_geo[:,:,:,0],y_true_geo[:,:,:,1]),tf.minimum(y_true_geo[:,:,:,2],y_true_geo[:,:,:,3])))
#     beta =1- tf.reduce_mean(labels)
#     loss = -1*tf.reduce_mean((beta * labels * tf.log(logits + eps)) + (1 - beta) * (1 - labels) * tf.log(1 - logits + eps))
#     tf.summary.scalar('classification_distance_balanced_sigmoid_crossentrop_loss', loss)
#     return loss


# def lovasz_grad(gt_sorted):
#     """
#     Computes gradient of the Lovasz extension w.r.t sorted errors
#     See Alg. 1 in paper
#     """
#     gts = tf.reduce_sum(gt_sorted)
#     intersection = gts - tf.cumsum(gt_sorted)
#     union = gts + tf.cumsum(1. - gt_sorted)
#     jaccard = 1. - intersection / union
#     jaccard = tf.concat((jaccard[0:1], jaccard[1:] - jaccard[:-1]), 0)
#     return jaccard



# def lovasz_hinge(y_true_cls, y_pred_cls, per_image=True, ignore=None):
#     """
#     Binary Lovasz hinge loss
#       logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
#       labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
#       per_image: compute the loss per image instead of per batch
#       ignore: void class id
#     """
#     labels=y_true_cls
#     logits=tf.nn.sigmoid(y_pred_cls)
#     if per_image:
#         def treat_image(log_lab):
#             log, lab = log_lab
#             log, lab = tf.expand_dims(log, 0), tf.expand_dims(lab, 0)
#             log, lab = flatten_binary_scores(log, lab, ignore)
#             return lovasz_hinge_flat(log, lab)
#         losses = tf.map_fn(treat_image, (logits, labels), dtype=tf.float32)
#         loss = tf.reduce_mean(losses)
#     else:
#         loss = lovasz_hinge_flat(*flatten_binary_scores(logits, labels, ignore))
#     return loss


# def lovasz_hinge_flat(logits, labels):
#     """
#     Binary Lovasz hinge loss
#       logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
#       labels: [P] Tensor, binary ground truth labels (0 or 1)
#       ignore: label to ignore
#     """

#     def compute_loss():
#         labelsf = tf.cast(labels, logits.dtype)
#         signs = 2. * labelsf - 1.
#         errors = 1. - logits * tf.stop_gradient(signs)
#         errors_sorted, perm = tf.nn.top_k(errors, k=tf.shape(errors)[0], name="descending_sort")
#         gt_sorted = tf.gather(labelsf, perm)
#         grad = lovasz_grad(gt_sorted)
#         loss = tf.tensordot(tf.nn.relu(errors_sorted), tf.stop_gradient(grad), 1, name="loss_non_void")
#         return loss

#     # deal with the void prediction case (only void pixels)
#     loss = tf.cond(tf.equal(tf.shape(logits)[0], 0),
#                    lambda: tf.reduce_sum(logits) * 0.,
#                    compute_loss,
#                    strict=True,
#                    name="loss"
#                    )
#     return loss


# def flatten_binary_scores(scores, labels, ignore=None):
#     """
#     Flattens predictions in the batch (binary case)
#     Remove labels equal to 'ignore'
#     """
#     scores = tf.reshape(scores, (-1,))
#     labels = tf.reshape(labels, (-1,))
#     if ignore is None:
#         return scores, labels
#     valid = tf.not_equal(labels, ignore)
#     vscores = tf.boolean_mask(scores, valid, name='valid_scores')
#     vlabels = tf.boolean_mask(labels, valid, name='valid_labels')
#     return vscores, vlabels



def loss(y_true_cls, y_pred_cls,
         y_true_geo, y_pred_geo,
         training_mask):
    '''
    define the loss used for training, contraning two part,
    the first part we use dice loss instead of weighted logloss,
    the second part is the iou loss defined in the paper
    :param y_true_cls: ground truth of text
    :param y_pred_cls: prediction os text
    :param y_true_geo: ground truth of geometry
    :param y_pred_geo: prediction of geometry
    :param training_mask: mask used in training, to ignore some text annotated by ###
    :return:
    '''
    classification_loss = dice_coefficient(y_true_cls, y_pred_cls,training_mask)
    # scale classification loss to match the iou loss part
    classification_loss *= 0.01

    # d1 -> top, d2->right, d3->bottom, d4->left
    d1_gt, d2_gt, d3_gt, d4_gt, theta_gt = tf.split(value=y_true_geo, num_or_size_splits=5, axis=3)
    d1_pred, d2_pred, d3_pred, d4_pred, theta_pred = tf.split(value=y_pred_geo, num_or_size_splits=5, axis=3)
    area_gt = (d1_gt + d3_gt) * (d2_gt + d4_gt)
    area_pred = (d1_pred + d3_pred) * (d2_pred + d4_pred)
    w_union = tf.minimum(d2_gt, d2_pred) + tf.minimum(d4_gt, d4_pred)
    h_union = tf.minimum(d1_gt, d1_pred) + tf.minimum(d3_gt, d3_pred)
    area_intersect = w_union * h_union
    area_union = area_gt + area_pred - area_intersect
    L_AABB = -tf.log((area_intersect + 1.0)/(area_union + 1.0))
    L_theta = 1 - tf.cos(theta_pred - theta_gt)
    tf.summary.scalar('geometry_AABB', tf.reduce_mean(L_AABB * y_true_cls * training_mask))
    tf.summary.scalar('geometry_theta', tf.reduce_mean(L_theta * y_true_cls * training_mask))
    L_g = L_AABB + 20 * L_theta

    return tf.reduce_mean(L_g * y_true_cls * training_mask) + classification_loss
