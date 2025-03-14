from strenum import StrEnum

class AggregationLevel(StrEnum):
    site = 'site' # not aggregated
    cluster = 'cluster'
    global_ = 'global'