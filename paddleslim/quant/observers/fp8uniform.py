# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import abc
from paddle.quantization.base_observer import BaseObserver
import paddle
from paddle.quantization.factory import ObserverFactory
class FP8UniformObserver(ObserverFactory):
    r"""
    It collects maximum absolute values of target tensor.
    Args:
        bit_length(int, optional): Number of bits to represent an quantized integer in binary.
        dtype(str, optional): The data type of input tensor.
        name (str, optional): This parameter is used by developers to print debugging information. \
            For details, please refer to :ref:`api_guide_Name`. Default is None.
    Examples:
       .. code-block:: python
            from paddle.quantization import QuantConfig
            from paddle.quantization.quanters import FakeQuanterWithAbsMaxObserver
            quanter = FakeQuanterWithAbsMaxObserver(moving_rate=0.99)
            q_config = QuantConfig(activation=quanter, weight=quanter)
    """

    def __init__(self):
        super(FP8UniformObserver, self).__init__()

    def _get_class(self):
        return FP8UniformObserverLayer

class FP8UniformObserverLayer(BaseObserver):
    """ This is the base class for a uniform quantization observer, which provides
    common functions for calculating the scale and zero-point used in uniform quantization.
    Uniform quantization maps floating point values to integers, where the scale determines
    the step size of the quantizer and the floating point zero is mapped to the zero-point,
    an integer value ensuring that zero is quantized without error.

    Args:
        quant_bits (int): The number of bits for quantization.
        sign (bool): Whether the quantized integer includes a sign.
        symmetric (bool): Whether it is symmetric quantization. the quantization is symmetric.
        In symmetric quantization, the range of floating point values is relaxed to be symmetric
        around zero and the zero-point is always 0.

    """

    def __init__(
        self,
        layer,
        quant_bits=8,
        ):
        super(FP8UniformObserverLayer, self).__init__()
        self._float8_type = "float8_e4m3fn"
        self._quant_bits = 8 
        self._min = None
        self._max = paddle.to_tensor(1e-7, dtype="float32")
        self._qmin = None
        self._qmax = None
        self._scale = None
        self._zero_point = None

    def qmin_qmax(self):
        """ Calculate the range of the quantized integer based on the specified
        float8_type."""
        if self._float8_type == "float8_e4m3fn":
            self._qmin = -448.0
            self._qmax = 448.0
        else:
            self._qmin = -57344.0
            self._qmax = +57344.0
        return self._qmin, self._qmax

    def min_value(self) -> float:
        """ The minimum value of floating-point numbers."""
        return self._min

    def max_value(self) -> float:
        """ The maximum value of floating-point numbers."""
        return self._max
    def cal_scales(self) -> float:
        """ Calculate the scales and zero points based on the min_value and max_value.
        """
        assert self.min_value() is not None and self.max_value() is not None
        _qmin, _qmax = self.qmin_qmax()
        # For one-sided distributions, the range (_min , _max ) is relaxed to include zero.
        # It is important to ensure that common operations like zero padding do not cause quantization errors.
        _min = min(self.min_value(), 0.)
        _max = max(self.max_value(), 0.)
        _abs_max = max(-_min, _max)
        self._scale = _qmax / _abs_max
        self._zero_point = 0
        return self._scale, self._zero_point

    def scales(self):
        """ Return output scales.
        """
        
        if self._scale is None:
            self.cal_thresholds()
        return self._scale
    def forward(self, inputs):
        """ Calculate forward pass.
        """
        self._min, self._max = self.cal_min_max(inputs)
        return inputs

    def cal_min_max(self, inputs):
        abs_max_val = paddle.max(paddle.abs(inputs.cast("float32")))
        abs_max_val = paddle.maximum(abs_max_val, self._max)
        return 0, abs_max_val
    def bit_length(self):
        """ Return the bit length of quantized data.
        """
        return self._quant_bits

    def quant_axis(self):
        """ Return quantization axis.
        """
        return -1
    def cal_thresholds(self):
        """ Compute thresholds for MAX function.
        """
        if self._scale is not None:
            self._zero_point = 0
            return
        self._scale, self._zero_point = self.cal_scales()
    def zero_points(self):
        """ Return output zero points.
        """
        if self._zero_point is None:
            self.cal_thresholds()
        return self._zero_point
