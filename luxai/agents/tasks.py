"""
Tasks for the units
"""

class BaseTask():
    # TODO: maybe an update method?
    # TODO: priority property?
    # TODO: get action method?
    def is_done(self, unit):
        raise NotImplementedError()


class GatherResourcesTask(BaseTask):
    def __init__(self, position):
        self.pos = position

    def is_done(self, unit):
        return not unit.get_cargo_space_left()


class GoToPositionTask(BaseTask):
    def __init__(self, position):
        self.pos = position

    def is_done(self, unit):
        return unit.pos.equals(self.pos)


class BuildCityTileTask(BaseTask):
    # TODO: this task probably needs more information to decide if it is done
    def __init__(self, position):
        self.pos = position

    def is_done(self, unit):
        return unit.pos.equals(self.pos)
