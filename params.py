integer_min = -256
integer_max = 255
integer_range = integer_max - integer_min + 1

max_list_len = 20
num_inputs = 3
num_examples = 5

max_program_len = 8
max_program_vars = max_program_len + num_inputs
state_len = max_program_vars + 1

type_vector_len = 2

embedding_size = 20

var_encoder_size = 56

dense_output_size = 256
dense_num_layers = 10
dense_growth_size = 56

dfs_max_width = 50

cab_beam_size = 100
cab_width = 10
cab_width_growth = 10
