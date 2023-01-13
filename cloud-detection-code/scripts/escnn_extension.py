"""
This module extends some commonly used modules from the escnn library
to be savable and loadable from files.
"""
import torch
from e2cnn import gspaces, nn
from scripts.invariant_gelu import GELU
from scripts.basic_blocks import NonGroupPoolingBlock

class InteroperableGroupPooling(nn.GroupPooling):
    """
    This module extends escnn.nn.GroupPooling to be easily saved + loaded from disk.
    """
    def get_save_dict(self, prefix=''):
        """
        This method saves all relevant properties of the InteroperableGroupPooling module
        in a dictionary.

        Arguments
        -----------
            prefix: string (optional)
                prefix for properties in dict

        Returns
        --------
            properties: dict
                contains properties to save to disk
        """
        properties = {}

        #Save vals from state dict
        state_dict = self.state_dict()
        for key, value in state_dict.items():
            properties[f"{prefix}.{key}"] = value

        #Save in/out type
        properties[f"{prefix}.in_channels"] = self.in_type.size

        return properties

    @classmethod
    def load_from_dict(cls, properties):
        """
        This method loads an InteroperableGroupPooling module from a state dictionary.

        Arguments
        ----------
            properties: dict
                dictionary of properties for module

        Returns
        --------
            gp_module: InteroperableGroupPooling
                module implementing those properties
        """
        #Get prefix and # of input channels
        in_channels = None
        for key in properties.keys():
            if '.' in key and key.split('.')[-1] == 'in_channels':
                in_channels = properties[key]
            elif key == 'in_channels':
                in_channels = properties[key]

        #Get input type and output type
        r2_act = gspaces.Rot2dOnR2(N=8)
        in_type = nn.FieldType(r2_act, round(in_channels/8)*[r2_act.regular_repr])

        gp_module = cls(in_type)
        for key in properties.keys():
            if key.split('.')[-1] != 'in_channels':
                setattr(gp_module, f"{key}.split('.')[-1]", properties[key])

        return gp_module

    def export(self):
        """
        This method exports the InteroperableGroupPooling module to a state dictionary.
        """
        self.eval()
        kernel_size = int(list(self._contiguous.keys())[0])

        return NonGroupPoolingBlock(kernel_size)

class InteroperableUpsample(nn.R2Upsampling):
    """
    This module extends escnn.nn.R2Upsampling to be easily saved + loaded from disk.
    """
    def get_save_dict(self, prefix=''):
        """
        This method saves all relevant properties of the InteroperableUpsample in a
        dictionary.

        Arguments
        -----------
            prefix: string (optional)
                prefix for properties in dict

        Returns
        --------
            properties: dict
                contains properties to save to disk
        """
        properties = {}

        #Save properties
        properties[f"{prefix}.channels"] = self.in_type.size
        properties[f"{prefix}.scale_factor"] = self._scale_factor
        properties[f"{prefix}.mode"] = self._mode
        properties[f"{prefix}.align_corners"] = self._align_corners

        return properties

    @classmethod
    def load_from_dict(cls, properties):
        """
        This method loads an InteroperableUpsample module from a state dictionary.

        Arguments
        ----------
            properties: dict
                dictionary of properties for module

        Returns
        --------
            ups_module: InteroperableUpsample
                module implementing those properties
        """
        #Get prefix
        prefix = ""
        for key in properties.keys():
            if key.split('.')[-1] == 'mode':
                prefix_keys = key.split('.')[0:-1]
                prefix = ".".join(prefix_keys)

        #Get properties
        channels = properties[f"{prefix}.channels"]
        scale_factor = properties[f"{prefix}.scale_factor"]
        mode = properties[f"{prefix}.mode"]
        align_corners = properties[f"{prefix}.align_corners"]

        #Get input type
        r2_act = gspaces.Rot2dOnR2(N=8)
        in_type = nn.FieldType(r2_act, round(channels/8)*[r2_act.regular_repr])

        #Initialize + return
        ups_module = InteroperableUpsample(in_type, scale_factor=scale_factor,
                mode=mode, align_corners=align_corners)

        return ups_module

class InteroperableMaxBlurPool(nn.PointwiseMaxPoolAntialiased):
    """
    This module extends escnn.nn.PointwiseMaxPoolAntialiased to be easily saved + loaded
    from disk.
    """
    def get_save_dict(self, prefix=''):
        """
        This method saves all relevant properties of the InteroperableMaxBlurPool to a
        state dictionary.

        Arguments
        -----------
            prefix: string (optional)
                prefix for properties in dict


        Returns
        ---------
            properties: dict
                contains properties to save to disk
        """
        properties = {}

        #Save properties
        properties[f'{prefix}.filter'] = self.filter
        properties[f'{prefix}.kernel_size'] = self.kernel_size[0] #Assume square kernel
        properties[f'{prefix}.padding'] = self.padding[0] #Assume square padding

        #Return
        return properties

    @classmethod
    def load_from_dict(cls, properties):
        """
        This method loads an InteroperableMaxBlurPool from a state dictionary containing
        a filter, input type, kernel size, and padding information.

        Arguments
        ----------
            properties: dict
                contains properties to load

        Returns
        --------
            pool_module: InteroperableMaxBlurPool
                module that shares properties from dictionary
        """
        #Get prefix
        prefix = ""
        for key in properties.keys():
            if key.split('.')[-1] == 'filter':
                prefix_keys = key.split('.')[0:-1]
                prefix = ".".join(prefix_keys)

        filter_tensor = properties[f'{prefix}.filter']
        kernel_size = properties[f'{prefix}.kernel_size']
        padding = properties[f'{prefix}.padding']

        #Get input type
        r2_act = gspaces.Rot2dOnR2(N=8)
        in_type = nn.FieldType(r2_act, round(filter_tensor.shape[0]/8)*[r2_act.regular_repr])

        pool_module = cls(in_type,
                kernel_size=kernel_size, padding=padding)
        pool_module.filter = filter_tensor

        return pool_module

class InteroperableReLU(nn.ReLU):
    """
    This module extends escnn.nn.ReLU to be easily saved + loaded from disk.
    """
    def get_save_dict(self, prefix=''):
        """
        This method saves an InteroperableReLU block to a state dictionary.

        Arguments
        -----------
            prefix: string (optional)
                prefix for properties

        Returns
        --------
            properties: dict
                dictionary with properties
        """
        properties = {}
        if self.in_type.representations[0].name == 'regular':
            properties[f"{prefix}.channels"] = round(self.in_type.size/8)
        else:
            properties[f"{prefix}.channels"] = self.in_type.size
        properties[f"{prefix}.inplace"] = self._inplace
        properties[f"{prefix}.is_regular"] = self.in_type.representations[0].name == 'regular'
        return properties

    @classmethod
    def load_from_dict(cls, properties):
        """
        This method loads an InteroperableReLU block from a state dictionary.

        Arguments
        ----------
            properties: dict
                properties for ReLU block

        Returns
        --------
            relu_module: InteroperableReLU
                ReLU block implementing properties
        """
        #Get properties
        channels = None
        inplace = None
        regular = False
        for key in properties.keys():
            if 'channels' in key:
                channels = properties[key]
            if 'inplace' in key:
                inplace = properties[key]
            if 'is_regular' in key:
                regular = properties[key]

        #Get input type
        r2_act = gspaces.Rot2dOnR2(N=8)
        if regular:
            in_type = nn.FieldType(r2_act, channels*[r2_act.regular_repr])
        else:
            in_type = nn.FieldType(r2_act, channels*[r2_act.trivial_repr])

        relu_module = InteroperableReLU(in_type, inplace=inplace)
        return relu_module

class InteroperableBatchNorm(nn.InnerBatchNorm):
    """
    This module extends escnn.nn.InnerBatchNorm to be easily saved + loaded from disk.
    """
    def get_save_dict(self, prefix=''):
        """
        This method saves an InteroperableBatchNorm block to a state dictionary.

        Arguments
        ----------
            prefix: string (optional)
                prefix for properties

        Returns
        --------
            properties: dict
                dictionary with properties
        """
        state_dict = self.state_dict()
        properties = {}
        for key, value in state_dict.items():
            properties[f"{prefix}.{key}"] = value

        properties[f"{prefix}.is_regular"] = self.in_type.representations[0].name == 'regular'
        return properties

    @classmethod
    def load_from_dict(cls, properties):
        """
        This method loads an InteroperableBatchNorm block from a state dictionary.

        Arguments
        ----------
            properties: dict
                contains properties to load

        Returns
        --------
            bn_module: InteroperableBatchNorm
                module that shares properties from dictionary
        """
        #Get input size (assume all channels contiguous)
        weight_shape = None
        regular = False
        for key in properties.keys():
            if '.' in key and key.split('.')[-1] == 'weight':
                weight_shape = properties[key].shape
            if 'is_regular' in key:
                regular = properties[key]

        #Get input type
        r2_act = gspaces.Rot2dOnR2(N=8)
        if regular:
            in_type = nn.FieldType(r2_act, weight_shape[0]*[r2_act.regular_repr])
        else:
            in_type = nn.FieldType(r2_act, weight_shape[0]*[r2_act.trivial_repr])

        #Initialize
        bn_module = cls(in_type)

        #Add properties (assume all channels contiguous)
        if regular:
            batch_norm = getattr(bn_module, 'batch_norm_[{}]'.format(8))
        else:
            batch_norm = getattr(bn_module, 'batch_norm_[{}]'.format(1))
        batch_norm.eval() #Update weights


        #Update properties (assuming all channels contiguous)
        for key in properties.keys():
            if '.' in key and 'is_regular' not in key:
                old_key = key
                key = key.split('.')[-1]
                #Only require grad for bias, weight
                requires_grad = key != "num_batches_tracked"
                requires_grad = requires_grad and "running" not in key
                requires_grad = requires_grad and "indices" not in key
                setattr(batch_norm, f"{key}", torch.nn.Parameter(data=properties[old_key],
                    requires_grad=requires_grad))

        #Update attribute and return
        if regular:
            setattr(bn_module, 'batch_norm_[{}]'.format(8), batch_norm)
        else:
            setattr(bn_module, 'batch_norm_[{}]'.format(1), batch_norm)

        return bn_module

class InteroperableR2Conv(nn.R2Conv):
    """
    This module extends escnn.nn.R2Conv to be easily saved + loaded from disk.
    """
    def get_save_dict(self, prefix=''):
        """
        This method gets a dictionary with key properties to save in a pickleable format.

        Arguments
        ----------
            prefix: string
                prefix for properties in dict

        Returns
        ---------
            properties: dict
                dictionary with properties to save
        """
        #Create properties dict and save weights/bias
        properties = {f"{prefix}.weights": self.weights, f"{prefix}.filter": self.filter}
        properties[f"{prefix}.bias"] = self.bias
        properties[f"{prefix}.expanded_bias"] = self.expanded_bias
        properties[f"{prefix}.is_regular"] = self.in_type.representations[0].name == 'regular'

        #Save padding info
        properties[f"{prefix}.padding"] = self.padding
        properties[f"{prefix}.padding_mode"] = self.padding_mode

        return properties


    @classmethod
    def load_from_dict(cls, properties):
        """
        This method loads an InteroperableR2Conv module from a dictionary of properties.

        Arguments
        ----------
            properties: dict
                dictionary with properties to load

        Returns
        --------
           eq_module: InteroperableR2Conv
                new module with loaded properties
        """
        #Filter size = channels out, channels in, kernel size
        filter_shape = None
        regular = False
        prefix = ""
        for key in properties.keys():
            if key.split('.')[-1] == 'filter':
                filter_shape = properties[key].shape
                prefix_keys = key.split('.')[0:-1]
                prefix = ".".join(prefix_keys)
            if 'is_regular' in key:
                regular = properties[key]

        #Get input type and output type
        r2_act = gspaces.Rot2dOnR2(N=8)
        if regular:
            in_type = nn.FieldType(r2_act, round(filter_shape[1]/8)*[r2_act.regular_repr])
        else:
            in_type = nn.FieldType(r2_act, filter_shape[1]*[r2_act.trivial_repr])

        #Out type is ALWAYS regular
        out_type = nn.FieldType(r2_act, round(filter_shape[0]/8)*[r2_act.regular_repr])

        #Get kernel size and padding
        kernel_size = filter_shape[2]
        padding = properties[f'{prefix}.padding']
        padding_mode = properties[f'{prefix}.padding_mode']

        #Create new InteroperableR2Conv
        eq_module = cls(in_type, out_type, kernel_size, padding=padding, padding_mode=padding_mode)

        #Update bias (if necessary)
        if properties[f'{prefix}.expanded_bias'] is not None:
            eq_module.bias = properties[f'{prefix}.bias'] 
            eq_module.expanded_bias = properties[f'{prefix}.expanded_bias']

        #Update filter + weights
        eq_module.filter = properties[f'{prefix}.filter']
        eq_module.weights = properties[f'{prefix}.weights']

        return eq_module

class InteroperableGELU(GELU):
    """
    This module extends escnn.nn.ReLU to be easily saved + loaded from disk.
    """
    def get_save_dict(self, prefix=''):
        """
        This method saves an InteroperableGELU block to a state dictionary.

        Arguments
        -----------
            prefix: string (optional)
                prefix for properties

        Returns
        --------
            properties: dict
                dictionary with properties
        """
        properties = {}
        if self.in_type.representations[0].name == 'regular':
            properties[f"{prefix}.channels"] = round(self.in_type.size/8)
        else:
            properties[f"{prefix}.channels"] = self.in_type.size
        properties[f"{prefix}.inplace"] = self._inplace
        properties[f"{prefix}.is_regular"] = self.in_type.representations[0].name == 'regular'
        return properties

    @classmethod
    def load_from_dict(cls, properties):
        """
        This method loads an InteroperableGELU block from a state dictionary.

        Arguments
        ----------
            properties: dict
                properties for GELU block

        Returns
        --------
            gelu_module: InteroperableGELU
                GELU block implementing properties
        """
        #Get properties
        channels = None
        inplace = None
        regular = False
        for key in properties.keys():
            if 'channels' in key:
                channels = properties[key]
            if 'inplace' in key:
                inplace = properties[key]
            if 'is_regular' in key:
                regular = properties[key]

        #Get input type
        r2_act = gspaces.Rot2dOnR2(N=8)
        if regular:
            in_type = nn.FieldType(r2_act, channels*[r2_act.regular_repr])
        else:
            in_type = nn.FieldType(r2_act, channels*[r2_act.trivial_repr])

        gelu_module = InteroperableGELU(in_type, inplace=inplace)
        return gelu_module


