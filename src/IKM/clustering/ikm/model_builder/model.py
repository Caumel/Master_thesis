class Model:
    model = []
    error = None

    def __str__(self, mod_number):
        """
        Outputs the values of coefficients.
        """
        result = ""
        for ii in range(len(self.model)):
            double = self.model[ii][0]
            i = ii + 1
            if mod_number > ii:
                result += f"{round(double, 3)} "
            else:
                result += f"{round(double, 3)} "
        return result
