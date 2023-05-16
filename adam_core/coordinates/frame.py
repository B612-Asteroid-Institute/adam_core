from quivr import StringField, Table


# TODO: Replace with DictionaryField or similar
#       Investigate whether this class is even necessary
class Frame(Table):
    name = StringField(nullable=False)
