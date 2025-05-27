import tensorflow as tf

def RIoU(center_label, size_label, heading_label, center_pred, size_pred, heading_pred):

	def cal_projection_area(origin, length, direction1, height, direction2, *points):
        points = [(p[0]-origin[0], p[1]-origin[1]) for p in points]
        proj_dil = tf.stack([p[0]*direction1[0]+p[1]*direction1[1] for p in points])
        proj_diw = tf.stack([p[0]*direction2[0]+p[1]*direction2[1] for p in points])
        max_dil = tf.reduce_max(proj_dil, axis=0)
        min_dil = tf.reduce_min(proj_dil, axis=0)
        max_diw = tf.reduce_max(proj_diw, axis=0)
        min_diw = tf.reduce_min(proj_diw, axis=0)
        length = length/2.
        height = height/2.

        intersection = tf.maximum(0., 
            tf.minimum(max_dil, length)-tf.maximum(min_dil, -length))*\
            tf.maximum(0., tf.minimum(max_diw, height)-tf.maximum(min_diw, -height))
        universal = tf.maximum(intersection, 
            (tf.maximum(max_dil, length)-tf.minimum(min_dil, -length))*\
            (tf.maximum(max_diw, height)-tf.minimum(min_diw, -height)))
        return intersection, universal
    x_gt, y_gt, z_gt = center_label[:,0], center_label[:,1], center_label[:,2]
    l_gt, w_gt, h_gt = size_label[:,0], size_label[:,1], size_label[:,2]
    r_gt = heading_label

    x_pred, y_pred, z_pred = center_pred[:,0], center_pred[:,1], center_pred[:,2]
    l_pred, w_pred, h_pred = size_pred[:,0], size_pred[:,1], size_pred[:,2]
    r_pred = heading_pred
    cos_gt = tf.cos(r_gt)
    sin_gt = tf.sin(r_gt)
    cos_pred = tf.cos(r_pred)
    sin_pred = tf.sin(r_pred)

    lcos_gt = l_gt*cos_gt/2.
    lsin_gt = l_gt*sin_gt/2.
    wsin_gt = -w_gt*sin_gt/2.
    wcos_gt = w_gt*cos_gt/2.
    lcos_pred = l_pred*cos_pred/2.
    lsin_pred = l_pred*sin_pred/2.
    wsin_pred = -w_pred*sin_pred/2.
    wcos_pred = w_pred*cos_pred/2.
    
    lu_gt = (x_gt-lcos_gt+wsin_gt, z_gt-lsin_gt+wcos_gt)
    ld_gt = (x_gt-lcos_gt-wsin_gt, z_gt-lsin_gt-wcos_gt)
    ru_gt = (x_gt+lcos_gt+wsin_gt, z_gt+lsin_gt+wcos_gt)
    rd_gt = (x_gt+lcos_gt-wsin_gt, z_gt+lsin_gt-wcos_gt)
    lu_pred = (x_pred-lcos_pred+wsin_pred, z_pred-lsin_pred+wcos_pred)
    ld_pred = (x_pred-lcos_pred-wsin_pred, z_pred-lsin_pred-wcos_pred)
    ru_pred = (x_pred+lcos_pred+wsin_pred, z_pred+lsin_pred+wcos_pred)
    rd_pred = (x_pred+lcos_pred-wsin_pred, z_pred+lsin_pred-wcos_pred)

    intersection1, universal1 = cal_projection_area((x_gt, z_gt), l_gt, (cos_gt, sin_gt), 
        w_gt, (-sin_gt, cos_gt), lu_pred, ld_pred, ru_pred, rd_pred)
    intersection2, universal2 = cal_projection_area((x_pred, z_pred), l_pred, (cos_pred, sin_pred), 
        w_pred, (-sin_pred, cos_pred), lu_gt, ld_gt, ru_gt, rd_gt)
    intersection = tf.minimum(intersection1, intersection2)*tf.abs(tf.cos(2.*(r_gt-r_pred)))
    #intersection = tf.minimum(intersection1, intersection2)
    universal = tf.maximum(universal1, universal2)

    v_gt = w_gt*l_gt
    v_pred = w_pred*l_pred
    union = tf.maximum(intersection, v_gt+v_pred-intersection)

    universal = tf.maximum(union, universal)

    mask_0 = tf.equal(union, 0.)
    mask_0 = tf.logical_or(mask_0, tf.equal(universal, 0.))
    mask_0 = tf.logical_not(mask_0)

    intersection = tf.boolean_mask(intersection, mask_0)
    union = tf.boolean_mask(union, mask_0)
    universal = tf.boolean_mask(universal, mask_0)
    giou = intersection/union
    #giou = intersection/union - (universal-union)/universal
    giou_loss = 1.-giou
    giou_loss = tf.reduce_mean(giou_loss)

	return giou_loss