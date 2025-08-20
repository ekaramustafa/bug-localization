from abc import ABC, abstractmethod

class BugLocalizationMethod(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def localize(self, bug_instance):
        pass