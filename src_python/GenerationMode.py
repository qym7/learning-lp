

class GenerationMode:

    def generate_information(self, selector, problem, path=None):
        pass

    def choose_vars_random(self, lps):
        pass

    def choose_constraints_random(self, lps, k):
        pass


class GenerationModeClassic(GenerationMode):

    def generate_information(self, selector, problem, path=None):
        return selector.generate_information_classic(problem, path)

    def choose_vars_random(self, lps):
        return lps.choose_vars_random()

    def choose_constraints_random(self, lps, k):
        return lps.choose_constraints_random_perturbation(k)


class GenerationModeMasterSlaveDiscreet(GenerationMode):

    def generate_information(self, selector, problem, path=None):
        return selector.generate_information_possible_vals(problem, path)

    def choose_vars_random(self, lps):
        return lps.choose_vars_random_general_cons()

    def choose_constraints_random(self, lps, k):
        return lps.choose_constraints_random_discreet()


class GenerationModeMasterSlaveContinuous(GenerationMode):

    def generate_information(self, selector, problem, path=None):
        return selector.generate_information_possible_vals(problem, path)

    def choose_vars_random(self, lps):
        return lps.choose_vars_random_general_cons()

    def choose_constraints_random(self, lps, k):
        return lps.choose_constraints_random_continuous()
