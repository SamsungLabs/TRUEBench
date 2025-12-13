class BaseAdaptor:
    def __init__(self, model_configs):
        raise NotImplementedError("This method should be implemented.")

    def terminate(self):
        raise NotImplementedError("This method should be implemented.")

    def inference(self, batch):
        """
        Perform inference on a batch of test cases.

        Parameters
        ----------
        batch : list
            List containing dictionary to have two parallel lists:

            - ``"input"`` (List[str])
                List of raw user/system prompts. Each element is a string.

            - ``"role"``  (List[Literal['system', 'user', 'assistant']])
                Corresponding role for each prompt. Length must be identical to
                ``batch["input"]``. Allowed values are ``'system'`` or ``'user'``.

            Both lists are interpreted position-wise; i.e. ``batch["input"][i]``
            is issued with role ``batch["role"][i]``.

        Returns
        -------
        result : list
            A list whose length equals ``len(batch)``.
            Result is an expanded version of each entry in the batch.
            The following fields are added to each entry.

            - ``"accumulated_conversations"``  (List[Dict[Literal['system', 'user', 'assistant'], str]]): the dialogue
            - ``"response"``       (List[str])   : Model-generated answer.
                                                   The list has the same length as the inputs.
                                                   When role is 'system', corresponding response is an empty string.
            - ``"think"``          (List[str])   : Internal reasoning.
            - ``"input_tokens"``   (List[int])   : Token count of the corresponding input string.
            - ``"think_tokens"``   (List[int])   : Token count of the ``think`` string.
            - ``"response_tokens"``(List[int])   : Token count of the ``response`` string.
            - ``"elapsed_time"``   (List[float]) : Seconds spent processing.

        """
        raise NotImplementedError("This method should be implemented.")

    def initialize_batch(self, batch):
        output_list = []
        for input in batch:
            # conserve other data (index, criterias, etc)
            output = input
            output["accumulated_conversations"] = []
            output["response"] = []
            output["think"] = []
            output["input_tokens"] = []
            output["think_tokens"] = []
            output["response_tokens"] = []
            output["elapsed_time"] = []
            output["role"] = input["role"]
            output["input"] = input["input"]
            output_list.append(output)
        return output_list
