# internal python imports
import warnings
from collections.abc import Iterable

# third party imports
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.layers as KL
import tensorflow.keras.initializers as KI

# local imports
import neurite as ne
import layers
import utils

# make directly available from vxm
ModelCheckpointParallel = ne.callbacks.ModelCheckpointParallel

class Combined_Unet(tf.keras.Model):
    """
    A unet architecture that builds off either an input keras model or input shape. Layer features 
    can be specified directly as a list of encoder and decoder features or as a single integer along
    with a number of unet levels. The default network features per layer (when no options are
    specified) are:

        encoder: [16, 32, 32, 32]
        decoder: [32, 32, 32, 32, 32, 16, 16]

    This network specifically does not subclass LoadableModel because it's meant to be a core,
    internal model for more complex networks, and is not meant to be saved/loaded independently.
    """

    def __init__(self,
                 inshape=None,
                 input_model_1=None,
                 input_model_2=None,
                 nb_features=None,
                 nb_levels=None,
                 max_pool=2,
                 feat_mult=1,
                 nb_conv_per_level=1,
                 do_res=False,
                 nb_upsample_skips=0,
                 hpy_input=None,
                 hyp_tensor=None,
                 kernel_initializer='he_normal',
                 name='unet'):
        """
        Parameters:
            inshape: Optional input tensor shape (including features). e.g. (192, 192, 192, 2).
            input_model: Optional input model that feeds directly into the unet before concatenation
            nb_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. 
                If None (default), the unet features are defined by the default config described in 
                the class documentation.
            nb_levels: Number of levels in unet. Only used when nb_features is an integer. 
                Default is None.
            feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. 
                Default is 1.
            nb_conv_per_level: Number of convolutions per unet level. Default is 1.
            nb_upsample_skips: Number of upsamples to skip in the decoder (to downsize the
                the output resolution). Default is 0.
            hyp_input: Hypernetwork input tensor. Enables HyperConvs if provided. Default is None.
            hyp_tensor: Hypernetwork final tensor. Enables HyperConvs if provided. Default is None.
            final_activation_function: Replace default activation function in final layer of unet.
            kernel_initializer: Initializer for the kernel weights matrix for conv layers. Default
                is 'he_normal'.
            name: Model name - also used as layer name prefix. Default is 'unet'.
        """

        # have the option of specifying input shape or input model

        if len(input_model_1.outputs) == 1:
            unet_input_1 = input_model_1.outputs[0]
        else:
            unet_input_1 = KL.concatenate(input_model_1.outputs, name='%s_input_bmode_concat' % name)

        if len(input_model_2.outputs) == 1:
            unet_input_2 = input_model_2.outputs[0]
        else:
            unet_input_2 = KL.concatenate(input_model_2.outputs, name='%s_input_rf_concat' % name)
        model_inputs = [input_model_1.inputs, input_model_2.inputs]

        # build feature list automatically
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError('must provide unet nb_levels if nb_features is an integer')
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
            nb_features = [
                np.repeat(feats[:-1], nb_conv_per_level),
                np.repeat(np.flip(feats), nb_conv_per_level)
            ]
        elif nb_levels is not None:
            raise ValueError('cannot use nb_levels if nb_features is not an integer')

        ndims = len(unet_input_1.shape) - 2
        assert ndims in (1, 2, 3), 'ndims should be one of 1, 2 or 3. found: %d' % ndims
        MaxPooling = getattr(KL, 'MaxPooling%dD' % ndims)

        # extract any surplus (full resolution) decoder convolutions
        enc_nf, dec_nf = nb_features
        nb_dec_convs = len(enc_nf)
        final_convs = dec_nf[nb_dec_convs:]
        dec_nf = dec_nf[:nb_dec_convs]
        nb_levels = int(nb_dec_convs / nb_conv_per_level) + 1

        if isinstance(max_pool, int):
            max_pool = [max_pool] * nb_levels


        # if final_activation_function is set, we need to build a utility that checks
        # which layer is truly the last, so we know not to apply the activation there
        activate = lambda lvl, c: True

        # configure bmode encoder (down-sampling path)
        enc_layers_bmode = []
        last_bmode = unet_input_1
        for level in range(nb_levels - 1):
            for conv in range(nb_conv_per_level):
                nf = enc_nf[level * nb_conv_per_level + conv]
                layer_name = '%s_enc_bmode_conv_%d_%d' % (name, level, conv)
                last_bmode = _conv_block(last_bmode, nf, name=layer_name, do_res=do_res, hyp_tensor=hyp_tensor,
                                   kernel_initializer=kernel_initializer)
            enc_layers_bmode.append(last_bmode)

            # temporarily use maxpool since downsampling doesn't exist in keras
            last_bmode = MaxPooling(max_pool[level], name='%s_enc_bmode_pooling_%d' % (name, level))(last_bmode)

        # configure bmode decoder (up-sampling path)
        for level in range(nb_levels - 1):
            real_level = nb_levels - level - 2
            for conv in range(nb_conv_per_level):
                nf = dec_nf[level * nb_conv_per_level + conv]
                layer_name = '%s_dec_bmode_conv_%d_%d' % (name, real_level, conv)
                last_bmode = _conv_block(last_bmode, nf, name=layer_name, do_res=do_res, hyp_tensor=hyp_tensor,
                                   include_activation=activate(level, conv),
                                   kernel_initializer=kernel_initializer)
            # upsample
            if level < (nb_levels - 1 - nb_upsample_skips):
                layer_name = '%s_dec_bmode_upsample_%d' % (name, real_level)
                last_bmode = _upsample_block(last_bmode, enc_layers_bmode.pop(), factor=max_pool[real_level],
                                       name=layer_name)
        

        # configure rf encoder (down-sampling path)
        enc_layers_rf = []
        last_rf = unet_input_2
        for level in range(nb_levels - 1):
            for conv in range(nb_conv_per_level):
                nf = enc_nf[level * nb_conv_per_level + conv]
                layer_name = '%s_enc_rf_conv_%d_%d' % (name, level, conv)
                last_rf = _conv_block(last_rf, nf, name=layer_name, do_res=do_res, hyp_tensor=hyp_tensor,
                                   kernel_initializer=kernel_initializer)
            enc_layers_rf.append(last_rf)

            # temporarily use maxpool since downsampling doesn't exist in keras
            last_rf = MaxPooling(max_pool[level], name='%s_enc_rf_pooling_%d' % (name, level))(last_rf)

        # configure bmode decoder (up-sampling path)
        for level in range(nb_levels - 1):
            real_level = nb_levels - level - 2
            for conv in range(nb_conv_per_level):
                nf = dec_nf[level * nb_conv_per_level + conv]
                layer_name = '%s_dec_rf_conv_%d_%d' % (name, real_level, conv)
                last_rf = _conv_block(last_rf, nf, name=layer_name, do_res=do_res, hyp_tensor=hyp_tensor,
                                   include_activation=activate(level, conv),
                                   kernel_initializer=kernel_initializer)
            # upsample
            if level < (nb_levels - 1 - nb_upsample_skips):
                layer_name = '%s_dec_rf_upsample_%d' % (name, real_level)
                last_rf = _upsample_block(last_rf, enc_layers_rf.pop(), factor=max_pool[real_level],
                                       name=layer_name)
                
        final_activate = lambda n: True

        # now we take care of any remaining convolutions
        for num, nf in enumerate(final_convs):
            if num == 0: 
                layer_name = '%s_dec_bmode_final_conv_%d' % (name, num)
                last_bmode = _conv_block(last_bmode, nf, name=layer_name, do_res=do_res, hyp_tensor=hyp_tensor,
                                   include_activation=activate(level, conv),
                                   kernel_initializer=kernel_initializer)
                
                layer_name = '%s_dec_rf_final_conv_%d' % (name, num)
                last_bmode = _conv_block(last_rf, nf, name=layer_name, do_res=do_res, hyp_tensor=hyp_tensor,
                                   include_activation=activate(level, conv),
                                   kernel_initializer=kernel_initializer)
               
                last = KL.concatenate([last_bmode, last_rf], name="%s_dec_rf_bmode_concat_%d" % (name, num))
            
            elif num == 1:
                layer_name = '%s_dec_final_conv_%d' % (name, num)
                last = _conv_block(last, nf, name=layer_name, hyp_tensor=hyp_tensor, 
                                include_activation=final_activate(num),
                                kernel_initializer=kernel_initializer)
                last = KL.concatenate([last_bmode, last], name="%s_dec_final_bmode_concat_%d" % (name, num))

            else:
                layer_name = '%s_dec_final_conv_%d' % (name, num)
                last = _conv_block(last, nf, name=layer_name, hyp_tensor=hyp_tensor, 
                                include_activation=final_activate(num),
                                kernel_initializer=kernel_initializer)

        # add the final activation function is set
        # if final_activation_function is not None:
        #     last = KL.Activation(final_activation_function, name='%s_final_activation' % name)(last)

        super().__init__(inputs=model_inputs, outputs=last, name=name)


class Vxm4D(ne.modelio.LoadableModel):
    """
    VoxelMorph network for (unsupervised) nonlinear registration between two images.
    """

    @ne.modelio.store_config_args
    def __init__(self,
                 inshape,
                 nb_unet_features=None,
                 nb_unet_levels=None,
                 unet_feat_mult=1,
                 nb_unet_conv_per_level=1,
                 int_steps=7,
                 svf_resolution=1,
                 int_resolution=2,
                 int_downsize=None,
                 bidir=False,
                 use_probs=False,
                 src_feats=1,
                 trg_feats=1,
                 unet_half_res=False,
                 fill_value=None,
                 reg_field='preintegrated',
                 name='vxm_4d'):
        """ 
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_unet_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. 
                If None (default), the unet features are defined by the default config described in 
                the unet class documentation.
            nb_unet_levels: Number of levels in unet. Only used when nb_unet_features is an integer. 
                Default is None.
            unet_feat_mult: Per-level feature multiplier. Only used when nb_unet_features is an 
                integer. Default is 1.
            nb_unet_conv_per_level: Number of convolutions per unet level. Default is 1.
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this 
                value is 0.
            svf_resolution: Resolution (relative voxel size) of the predicted SVF.
                Default is 1.
            int_resolution: Resolution (relative voxel size) of the flow field during
                vector integration. Default is 2.
            int_downsize: Deprecated - use int_resolution instead.
            bidir: Enable bidirectional cost function. Default is False.
            use_probs: Use probabilities in flow field. Default is False.
            src_feats: Number of source image features. Default is 1.
            trg_feats: Number of target image features. Default is 1.
            unet_half_res: Deprecated - use svf_resolution instead.
            input_model: Model to replace default input layer before concatenation. Default is None.
            reg_field: Field to regularize in the loss. Options are 'svf' to return the
                SVF predicted by the Unet, 'preintegrated' to return the SVF that's been
                rescaled for vector-integration (default), 'postintegrated' to return the
                rescaled vector-integrated field, and 'warp' to return the final, full-res warp.
            name: Model name - also used as layer name prefix. Default is 'vxm_dense'.
        """

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2 or 3. found: %d' % ndims

        # No input model is passed, so configure default input layers
        source = tf.keras.Input(shape=(*inshape, src_feats), name='%s_source_bmode_input' % name)
        target = tf.keras.Input(shape=(*inshape, trg_feats), name='%s_target_bmode_input' % name)
        source_rf = tf.keras.Input(shape=(*inshape, src_feats), name='%s_source_rf_input' % name)
        target_rf = tf.keras.Input(shape=(*inshape, src_feats), name='%s_target_rf_input' % name)
        input_model_bmode = tf.keras.Model(inputs=[source, target], outputs=[source, target])
        input_model_rf = tf.keras.Model(inputs=[source_rf, target_rf], outputs=[source_rf, target_rf])

        inputs = (source, target, source_rf, target_rf)

        if int_downsize is not None:
            warnings.warn('int_downsize is deprecated, use the int_resolution parameter.')
            int_resolution = int_downsize

        # compute number of upsampling skips in the decoder (to downsize the predicted field)
        if unet_half_res:
            warnings.warn('unet_half_res is deprecated, use the svf_resolution parameter.')
            svf_resolution = 2

        nb_upsample_skips = int(np.floor(np.log(svf_resolution) / np.log(2)))

        # build core unet model and grab inputs
        unet_model = Combined_Unet(
            input_model_1=input_model_bmode,
            input_model_2=input_model_rf, 
            nb_features=nb_unet_features, # x, y, time, 2 (bmode, rf)
            nb_levels=nb_unet_levels,
            feat_mult=unet_feat_mult,
            nb_conv_per_level=nb_unet_conv_per_level,
            nb_upsample_skips=nb_upsample_skips,
            name='%s_unet' % name,
        )

        # transform unet output into a flow field
        Conv = getattr(KL, 'Conv%dD' % ndims)
        flow_mean = Conv(ndims, kernel_size=3, padding='same',
                         kernel_initializer=KI.RandomNormal(mean=0.0, stddev=1e-5),
                         name='%s_flow' % name)(unet_model.output)

        # optionally include probabilities
        if use_probs:
            # initialize the velocity variance very low, to start stable
            flow_logsigma = Conv(ndims, kernel_size=3, padding='same',
                                 kernel_initializer=KI.RandomNormal(mean=0.0, stddev=1e-10),
                                 bias_initializer=KI.Constant(value=-10),
                                 name='%s_log_sigma' % name)(unet_model.output)
            flow_params = KL.concatenate([flow_mean, flow_logsigma], name='%s_prob_concat' % name)
            flow_inputs = [flow_mean, flow_logsigma]
            flow = ne.layers.SampleNormalLogVar(name='%s_z_sample' % name)(flow_inputs)
        else:
            flow = flow_mean

        # rescale field to target svf resolution
        pre_svf_size = np.array(flow.shape[1:-1])
        svf_size = np.array([np.round(dim / svf_resolution) for dim in inshape])
        if not np.array_equal(pre_svf_size, svf_size):
            rescale_factor = svf_size[0] / pre_svf_size[0]
            flow = layers.RescaleTransform(rescale_factor, name=f'{name}_svf_resize')(flow)

        # cache svf
        svf = flow

        # rescale field to target integration resolution
        if int_steps > 0 and int_resolution > 1:
            int_size = np.array([np.round(dim / int_resolution) for dim in inshape])
            if not np.array_equal(svf_size, int_size):
                rescale_factor = int_size[0] / svf_size[0]
                flow = layers.RescaleTransform(rescale_factor, name=f'{name}_flow_resize')(flow)

        # cache pre-integrated flow field
        preint_flow = flow

        # optionally negate flow for bidirectional model
        pos_flow = flow
        if bidir:
            neg_flow = ne.layers.Negate(name='%s_neg_flow' % name)(flow)

        # integrate to produce diffeomorphic warp (i.e. treat flow as a stationary velocity field)
        if int_steps > 0:
            pos_flow = layers.VecInt(method='ss',
                                     name='%s_flow_int' % name,
                                     int_steps=int_steps)(pos_flow)
            if bidir:
                neg_flow = layers.VecInt(method='ss',
                                         name='%s_neg_flow_int' % name,
                                         int_steps=int_steps)(neg_flow)

        # cache the intgrated flow field
        postint_flow = pos_flow

        # resize to final resolution
        if int_steps > 0 and int_resolution > 1:
            rescale_factor = inshape[0] / int_size[0]
            pos_flow = layers.RescaleTransform(rescale_factor, name='%s_diffflow' % name)(pos_flow)
            if bidir:
                neg_flow = layers.RescaleTransform(rescale_factor,
                                                   name='%s_neg_diffflow' % name)(neg_flow)

        # warp image with flow field
        y_source = layers.SpatialTransformer(
            interp_method='linear',
            indexing='ij',
            fill_value=fill_value,
            name='%s_transformer' % name)([source, pos_flow])

        if bidir:
            st_inputs = [target, neg_flow]
            y_target = layers.SpatialTransformer(interp_method='linear',
                                                 indexing='ij',
                                                 fill_value=fill_value,
                                                 name='%s_neg_transformer' % name)(st_inputs)

        # initialize the keras model
        outputs = [y_source, y_target] if bidir else [y_source]

        # determine regularization output
        reg_field = reg_field.lower()
        if use_probs:
            # compute loss on flow probabilities
            outputs.append(flow_params)
        elif reg_field == 'svf':
            # regularize the immediate, predicted SVF
            outputs.append(svf)
        elif reg_field == 'preintegrated':
            # regularize the rescaled, pre-integrated SVF
            outputs.append(preint_flow)
        elif reg_field == 'postintegrated':
            # regularize the rescaled, integrated field
            outputs.append(postint_flow)
        elif reg_field == 'warp':
            # regularize the final, full-resolution deformation field
            outputs.append(pos_flow)
        else:
            raise ValueError(f'Unknown option "{reg_field}" for reg_field.')

        super().__init__(name=name, inputs=inputs, outputs=outputs)

        # cache pointers to layers and tensors for future reference
        self.references = ne.modelio.LoadableModel.ReferenceContainer()
        self.references.unet_model = unet_model
        self.references.source = source
        self.references.target = target
        self.references.svf = svf
        self.references.preint_flow = preint_flow
        self.references.postint_flow = postint_flow
        self.references.pos_flow = pos_flow
        self.references.neg_flow = neg_flow if bidir else None
        self.references.y_source = y_source
        self.references.y_target = y_target if bidir else None

    def get_registration_model(self):
        """
        Returns a reconfigured model to predict only the final transform.
        """
        return tf.keras.Model(self.inputs, self.references.pos_flow)

    def register(self, src, trg):
        """
        Predicts the transform from src to trg tensors.
        """
        return self.get_registration_model().predict([src, trg])

    def apply_transform(self, src, trg, img, interp_method='linear'):
        """
        Predicts the transform from src to trg and applies it to the img tensor.
        """
        warp_model = self.get_registration_model()
        img_input = tf.keras.Input(shape=img.shape[1:])
        st_input = [img_input, warp_model.output]
        y_img = layers.SpatialTransformer(interp_method=interp_method)(st_input)
        return tf.keras.Model(warp_model.inputs + [img_input], y_img).predict([src, trg, img])


#__________________________________________________________________________________________________


def _conv_block(x, nfeat, strides=1, name=None, do_res=False, hyp_tensor=None,
                include_activation=True, kernel_initializer='he_normal'):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """
    ndims = len(x.shape) - 2
    assert ndims in (1, 2, 3), 'ndims should be one of 1, 2, or 3. found: %d' % ndims

    extra_conv_params = {}

    Conv = getattr(KL, 'Conv%dD' % ndims)
    extra_conv_params['kernel_initializer'] = kernel_initializer
    conv_inputs = x

    convolved = Conv(nfeat, kernel_size=3, padding='same',
                     strides=strides, name=name, **extra_conv_params)(conv_inputs)

    if do_res:
        # assert nfeat == x.get_shape()[-1], 'for residual number of features should be constant'
        add_layer = x
        print('note: this is a weird thing to do, since its not really residual training anymore')
        if nfeat != x.shape.as_list()[-1]:
            add_layer = Conv(nfeat, kernel_size=3, padding='same',
                             name='resfix_' + name, **extra_conv_params)(conv_inputs)
        convolved = KL.Lambda(lambda x: x[0] + x[1])([add_layer, convolved])

    if include_activation:
        name = name + '_activation' if name else None
        convolved = KL.LeakyReLU(0.2, name=name)(convolved)

    return convolved


def _upsample_block(x, connection, factor=2, name=None):
    """
    Specific upsampling and concatenation layer for unet.
    """
    ndims = len(x.shape) - 2
    assert ndims in (1, 2, 3), 'ndims should be one of 1, 2, or 3. found: %d' % ndims
    UpSampling = getattr(KL, 'UpSampling%dD' % ndims)

    size = (factor,) * ndims if ndims > 1 else factor
    upsampled = UpSampling(size=size, name=name)(x)
    name = name + '_concat' if name else None
    return KL.concatenate([upsampled, connection], name=name)