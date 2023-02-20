class Nn:
    def __init__(self, history = None, data=None):
         self._history = history
         self._data = data
      
    # getter method history fitting
    def get_history(self):
        return self._history
      
    # setter method history fitting
    def set_history(self, x):
        self._history = x

    def get_data(self):
        return self._data
      
    # setter method history fitting
    def set_data(self, x):
        self._data = x