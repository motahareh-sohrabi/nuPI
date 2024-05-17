import abc

import cooper


class BaseProblem(cooper.ConstrainedMinimizationProblem, abc.ABC):
    has_dual_variables: bool

    @abc.abstractmethod
    def compute_cmp_state(self) -> cooper.CMPState:
        pass

    @abc.abstractmethod
    def dual_parameter_groups(self):
        pass

    @abc.abstractmethod
    def create_multiplier(self):
        # This class may be implemented as a dummy function for CMPs that do not
        # require multipliers. Forcing implementations to define this function
        # for sanity-checking.
        pass
