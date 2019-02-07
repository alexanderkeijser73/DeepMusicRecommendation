
# DATASET FILE LOCATIONS
data_path = '/Users/Alexander/Documents/master_ai/Project Music Recommendation/DeepMusicRecommendation/data'
user_item_matrix = 'wmf/user_item_matrix.pkl'
wmf_item2i = 'wmf/index_dicts.pkl'
wmf_user2i = 'wmf/index_dicts.pkl'
track_to_song = 'wmf/track_to_song.pkl'
item_factors = 'wmf/item_wmf_50.pkl'
user_factors = 'wmf/user_wmf_50.pkl'
track_id_to_info = 'song_metadata/track_id_to_info.pkl'

# Training params
batch_size =16
valid_batch_size = 256
learning_rate = 0.03
num_epochs = 25
max_norm = 5.0

# Misc params
checkpoint_path = 'checkpoints_cnn'
summary_path = '../tensorboard_summaries/'
print_every = 10
save_every = 250
validate_every = 10
prev_checkpoint_path = None