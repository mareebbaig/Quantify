class QuantizerManager:
    """
    Manager object shared across all quantizers.
    Used for global coordination, such as forcing re-calibration 
    or tracking global quantization statistics.
    """
    def __init__(self):
        # Global flag to force all quantizers to re-run their search/calibration
        self.force_recalibration = False
        # Registry to keep track of all active quantizer instances
        self.quantizers = []

    def register_quantizer(self, quantizer):
        """Registers a quantizer instance with the manager."""
        if quantizer not in self.quantizers:
            self.quantizers.append(quantizer)

    def trigger_global_recalibration(self):
        """Sets the flag to force all quantizers to re-calibrate on next forward."""
        self.force_recalibration = True

    def reset_global_flag(self):
        """Resets the global recalibration flag."""
        self.force_recalibration = False

# The single shared reference for the entire framework
quantizer_manager = QuantizerManager()
