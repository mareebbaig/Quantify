class QuantizerManager:
    """
    Manager object for coordinating quantizer instances within a single model/run.
    Used for global coordination, such as forcing re-calibration 
    or tracking global quantization statistics.
    
    Note: This class is no longer a global singleton. Instantiate it explicitly
    per training run or model to avoid state leakage across experiments or DDP ranks.
    """
    def __init__(self):
        # Global flag to force all quantizers to re-run their search/calibration
        self.force_recalibration = False
        self.quantization_is_enabled_globally = True
        self.quantization_start_gap = 0
        # Registry to keep track of all active quantizer instances {id: quantizer}
        self.quantizers = {}
        # Counter to generate unique identifiers
        self._id_counter = 0
        self._inference_sequence_id_counter = 0

    def stop_quantization_for_n_inferences(self, n):
        for quant in self.quantizers.values():
            quant.inference_counter = -n

    def set_annealing_for_n_inferences(self, n):
        alpha_step = 1.0/n
        for quant in self.quantizers.values():
            quant.annealing_alpha = 0
            quant.annealing_alpha_step = alpha_step

    def register_quantizer(self, quantizer):
        """
        Registers a quantizer instance with the manager and assigns it a unique ID.
        """
        if quantizer in self.quantizers.values():
            return

        qid = f"quant_{self._id_counter}"
        self.quantizers[qid] = quantizer
        
        # Assign the unique ID back to the quantizer object for easy reference
        quantizer.quant_id = qid
        quantizer.inference_sequence_id = -1

        self._id_counter += 1

    def get_inference_sequence_id(self):
        self._inference_sequence_id_counter += 1
        return self._inference_sequence_id_counter - 1

    def trigger_global_recalibration(self):
        """Sets the flag to force all quantizers to re-calibrate on next forward."""
        self.force_recalibration = True

    def reset_global_flag(self):
        """Resets the global recalibration flag."""
        self.force_recalibration = False
