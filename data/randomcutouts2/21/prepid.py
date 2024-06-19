
#change ID names in txt

# Open the input text file
with open('filenames.txt', 'r') as input_file:
    # Read the IDs from the file
    ids = input_file.read().split()

# Add single quotes around each ID
quoted_ids = ["'" + id + "'" for id in ids]

# Join the quoted IDs into a single string
quoted_ids_string = ' '.join(quoted_ids)

# Write the quoted IDs to the output text file
with open('filenames_output.txt', 'w') as output_file:
    output_file.write(quoted_ids_string)
    
    