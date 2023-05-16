from quivr import StringField, Table


# TODO: Replace with DictionaryField or similar
#       Investigate whether this class is even necessary
class Origin(Table):
    code = StringField(nullable=False)
