
class GameInfo():
    """
    Class to store all the relevant information of the game for taking decisions
    """
    def __init__(self):
        self.resource_tiles = None
        self.empty_tiles = None
        self.available_workers = None
        self.non_available_workers = None
        self.city_tile_positions = None
        self.opponent_city_tile_positions = None
        self.obstacles = None
        self.is_night = False