from keras.models import Model
from keras.models import Input
from tensorflow.keras.optimizers import Adam
from discriminator import define_discriminator
from generator import define_generator

'''Generator-A Composite Model

Only Generator-A weights are trainable and weights for other models and not trainable.

Adversarial Loss: Domain-B -> Generator-A -> Domain-A -> Discriminator-A -> [real/fake]
Identity Loss: Domain-A -> Generator-A -> Domain-A
Forward Cycle Loss: Domain-B -> Generator-A -> Domain-A -> Generator-B -> Domain-B
Backward Cycle Loss: Domain-A -> Generator-B -> Domain-B -> Generator-A -> Domain-A
Inputs: Domain-B, Domain-A
Outputs: Real, Domain-A, Domain-B, Domain-A'''
'''Same but oposite for G-B'''


# define a composite model for updating generators by adversarial and cycle loss
def define_composite_model(g_model_1, d_model, g_model_2, image_shape):
	# ensure the model we're updating is trainable
	g_model_1.trainable = True
	# mark discriminator as not trainable
	d_model.trainable = False
	# mark other generator model as not trainable
	g_model_2.trainable = False
	# discriminator element
	input_gen = Input(shape=image_shape)
	gen1_out = g_model_1(input_gen)
	output_d = d_model(gen1_out)
	# identity element
	input_id = Input(shape=image_shape)
	output_id = g_model_1(input_id)
	# forward cycle
	output_f = g_model_2(gen1_out)
	# backward cycle
	gen2_out = g_model_2(input_id)
	output_b = g_model_1(gen2_out)
	# define model graph
	model = Model([input_gen, input_id], [output_d, output_id, output_f, output_b])
	# define optimization algorithm configuration
	opt = Adam(lr=0.0002, beta_1=0.5)
	# compile model with weighting of least squares loss and L1 loss
	model.compile(loss=['mse', 'mae', 'mae', 'mae'], loss_weights=[1, 5, 10, 10], optimizer=opt)
	return model


# # input shape
# image_shape = (256,256,3)
# # generator: A -> B
# g_model_AtoB = define_generator(image_shape)
# # generator: B -> A
# g_model_BtoA = define_generator(image_shape)
# # discriminator: A -> [real/fake]
# d_model_A = define_discriminator(image_shape)
# # discriminator: B -> [real/fake]
# d_model_B = define_discriminator(image_shape)

# # composite: A -> B -> [real/fake, A]
# c_model_AtoBtoA = define_composite_model(g_model_AtoB, d_model_B, g_model_BtoA, image_shape)
# # composite: B -> A -> [real/fake, B]
# c_model_BtoAtoB = define_composite_model(g_model_BtoA, d_model_A, g_model_AtoB, image_shape)