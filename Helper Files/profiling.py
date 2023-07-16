import inference

import inference  # Import your inference.py module
import cProfile

cProfile.runctx('inference.main()', globals(), locals(), filename='profile_data.prof')
