"""
Tasks for the units
"""

class BaseTask():
    # TODO: maybe an update method?
    # TODO: priority property?
    def __init__(self, pos):
        self.pos = pos

    def is_done(self, unit):
        raise NotImplementedError()

    def get_action(self, unit):
        return self.move_to_position(unit)

    def move_to_position(self, unit):
        return unit.move(unit.pos.direction_to(self.pos))


class GatherResourcesTask(BaseTask):
    def is_done(self, unit):
        return not unit.get_cargo_space_left()


class GoToPositionTask(BaseTask):
    def is_done(self, unit):
        return unit.pos.equals(self.pos)


class BuildCityTileTask(BaseTask):
    def __init__(self, pos):
        super().__init__(pos)
        self.is_city_built = False

    # TODO: this task probably needs more information to decide if it is done
    def is_done(self, unit):
        return unit.pos.equals(self.pos) and self.is_city_built

    def get_action(self, unit):
        if not unit.pos.equals(self.pos):
            return self.move_to_position(unit)
        else:
            self.is_city_built = True
            return unit.build_city()
