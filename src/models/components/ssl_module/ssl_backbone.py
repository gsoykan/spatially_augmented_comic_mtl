from enum import Enum


class SSLBackbone(str, Enum):
    CORINFOMAX = 'CORINFOMAX'
    CORINFOMAX_FINETUNE = 'CORINFOMAX_FINETUNE'
    CORINFOMAX_PROJ_1 = 'CORINFOMAX_PROJ_1'
    CORINFOMAX_PROJ_1_FINETUNE = 'CORINFOMAX_PROJ_1_FINETUNE'
    SIM_CLR = 'SIM_CLR'
    SIM_CLR_FINETUNE = 'SIM_CLR_FINETUNE'
    # when deeper proj. head is used -> keep the top - 1 linear layer
    SIM_CLR_DEEPER_LAST = 'SIM_CLR_DEEPER_LAST'
    # keeps all of the sim_clr with the projection head...
    SIM_CLR_KEEP_ALL = 'SIM_CLR_KEEP_ALL'
    # keeps all of the sim_clr with the projection head but fine-tunes last linear layer
    SIM_CLR_KEEP_ALL_ACTIVATE_LAST_LINEAR = 'SIM_CLR_KEEP_ALL_ACTIVATE_LAST_LINEAR'
