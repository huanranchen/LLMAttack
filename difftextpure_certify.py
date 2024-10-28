from models import Vicuna15
from models.SEDD import DiffTextPureUniform

model = DiffTextPureUniform(Vicuna15())
model.certify_given_pA(1)
