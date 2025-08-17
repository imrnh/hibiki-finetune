# import sentencepiece as spm
# from langdetect import detect
# from tqdm.auto import tqdm

# # Load model
# sp = spm.SentencePieceProcessor()
# sp.load("tokenizer_spm_48k_multi6_2.model")

# print(f"Vocab size: {sp.get_piece_size()}")

# langs_found = set()
# for i in tqdm(range(sp.get_piece_size())):
#     token = sp.id_to_piece(i)
#     if not token.startswith("<"):  # skip special tokens
#         try:
#             langs_found.add(detect(token))
#         except:
#             pass

# print("Detected languages:", langs_found)

# while True:
#     try:
#         idx = input("Enter vocab index (or 'q' to quit): ")
#         if idx.lower() == 'q':
#             break
#         idx = int(idx)
#         if 0 <= idx < sp.get_piece_size():
#             piece = sp.id_to_piece(idx)
#             score = sp.get_score(idx)
#             print(f"Index {idx}: Token = {piece}, Score = {score}")
#         else:
#             print("Index out of range.")
#     except ValueError:
#         print("Please enter a valid integer.")


import sentencepiece as spm

# Load the model
sp = spm.SentencePieceProcessor()
sp.load("tokenizer_spm_48k_multi6_2.model")

# Write vocab to file
with open("vocab_dump.txt", "w", encoding="utf-8") as f:
    for i in range(sp.get_piece_size()):
        piece = sp.id_to_piece(i)
        score = sp.get_score(i)
        f.write(f"{i}\t{piece}\t{score}\n")

print("Vocab written to vocab_dump.txt")
