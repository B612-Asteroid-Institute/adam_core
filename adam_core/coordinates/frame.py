import numpy as np
import pyarrow.compute as pc
from quivr import StringField, Table


# TODO: Replace with DictionaryField or similar
#       Investigate whether this class is even necessary
class Frame(Table):
    name = StringField(nullable=False)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, (str, np.ndarray)):
            return pc.all(pc.equal(self.name, other)).as_py()
        elif isinstance(other, Frame):
            return pc.all(pc.equal(self.name, other.name)).as_py()
        else:
            raise NotImplementedError(f"Cannot compare Frame to type: {type(other)}")
