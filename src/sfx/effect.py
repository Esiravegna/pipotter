class Effect(object):
    """
    Just a base object that provides an interfase for the controller to run them.
    The rationale is that each detected spell should run a single effect, that can be lights, audio, a combination of them,
     or whatnot.
    Mind the submodules under this folder for that.
    """

    name = "BASE_OBJECT"  # The name the Effect has, for identifying purposes

    def run(self, *args, **kwargs):
        """
        The must-be-implemented method for all the effects being created
        :param args: generic list of arguments that may be needed
        :param kwargs: generic dict of arguments that may be needed
        :return:
        """
        raise NotImplemented

    def __str__(self):
        """
        Returns the name of the class
        :return: str, the name
        """
        return "Effect object - {}".format(self.name)
