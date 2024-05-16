from paddle.nn import functional as F
class FP8QuantedLinear(FP8ConvertibleQuantedLayer):
    """
    The computational logic of QuantizedLinear is the same as Linear.
    The only difference is that its inputs are all fake quantized.
    """

    def __init__(self, layer: Layer, q_config):
        super().__init__()
        # For Linear
        self.weight = layer.weight
        self.bias = layer.bias
        self.name = layer.name
        # For FakeQuant

        self.weight_quanter = None
        self.activation_quanter = None
        if q_config.weight is not None:
            self.weight_quanter = q_config.weight._instance(layer)
        if q_config.activation is not None:
            self.activation_quanter = q_config.activation._instance(layer)

    def forward(self, input):
        quant_input = input
        quant_weight = self.weight
        if self.activation_quanter is not None:
            quant_input = self.activation_quanter(input)
        if self.weight_quanter is not None:
            quant_weight = self.weight_quanter(self.weight)
        return self._linear_forward(quant_input, quant_weight)

    def _linear_forward(self, input, weight):
        out = F.linear(x=input, weight=weight, bias=self.bias, name=self.name)
        return out

    def weights_to_quanters(self):
        return [('weight', 'weight_quanter')]

    def activation_quanters(self):
        return ['activation_quanter']

