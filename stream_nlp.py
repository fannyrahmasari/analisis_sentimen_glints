import pickle
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer

# Daftar kata sentimen
negative_words = ['kurang', 'tidak', 'buruk', 'jelek', 'gagal', 'menipu', 'mengecewakan', 'lelet', 'sulit', 'parah', 'penipuan', 'lambat']
neutral_words = ['cukup', 'biasa', 'standar', 'lumayan', 'oke', 'semoga', 'ku kasih', 'rata-rata', 'moderasi', 'sementara', 'kasih']
positive_words = ['baik', 'bagus', 'memuaskan', 'cepat', 'hebat', 'terbaik', 'puas', 'sempurna', 'luar biasa', 'mudah']

# Fungsi tambahan untuk mendeteksi kata sentimen
def detect_sentiment_keywords(text):
    text_tokens = text.lower().split()
    contains_negative = any(word in text_tokens for word in negative_words)
    contains_neutral = any(word in text_tokens for word in neutral_words)
    contains_positive = any(word in text_tokens for word in positive_words)
    
    if contains_negative:
        return "Negatif"
    elif contains_positive:
        return "Positif"
    elif contains_neutral:
        return "Netral"
    else:
        return None

# Memuat vocabulary yang sudah disimpan
with open("features_tf-idf.sav", "rb") as file:
    vocab = pickle.load(file)

# Membuat TfidfVectorizer dengan vocabulary yang sudah disesuaikan
vec_TF_IDF_loaded = TfidfVectorizer(
    ngram_range=(1, 2), 
    vocabulary=vocab, 
    decode_error="ignore", 
    use_idf=True, 
    max_features=3000
)

# Fitting dengan dokumen kosong agar vectorizer siap digunakan
vec_TF_IDF_loaded.fit([""])  # Fit dengan dokumen kosong untuk memastikan vectorizer siap

# Memuat model sentimen
with open("model_sentimen_glints.sav", "rb") as file:
    model = pickle.load(file)  # Langsung memuat model, tanpa mengambil dari dictionary

# Judul Aplikasi
st.title("Prediksi Sentimen Ulasan")

# Input dari pengguna (menggunakan text_area agar lebih besar)
user_input = st.text_area("Masukkan komentar ulasan:", height=200)

if st.button("Prediksi Sentimen"):
    if user_input.strip() == "":
        st.warning("Harap masukkan komentar terlebih dahulu!")
    else:
        # Transformasi data input
        try:
            data_input_transformed = vec_TF_IDF_loaded.transform([user_input])

            # Mengonversi sparse matrix ke array yang lebih mudah dikelola
            data_input_dense = data_input_transformed.toarray()

            # Memastikan jumlah fitur sesuai dengan yang diharapkan (3000 fitur)
            if data_input_dense.shape[1] < 3000:
                padding = np.zeros((data_input_dense.shape[0], 3000 - data_input_dense.shape[1]))
                data_input_dense = np.hstack([data_input_dense, padding])
            elif data_input_dense.shape[1] > 3000:
                data_input_dense = data_input_dense[:, :3000]

            # Prediksi menggunakan model
            hasil_model = model.predict(data_input_dense)

            # Menampilkan hasil berdasarkan kata kunci atau model
            hasil_keywords = detect_sentiment_keywords(user_input)

            # Gabungkan hasil berdasarkan kata kunci dan model
            if hasil_keywords:
                # Menampilkan hasil berdasarkan kata kunci dengan warna yang sesuai
                if hasil_keywords == "Negatif":
                    st.error(f"Hasil berdasarkan kata kunci: **{hasil_keywords}**")
                elif hasil_keywords == "Netral":
                    st.info(f"Hasil berdasarkan kata kunci: **{hasil_keywords}**")
                elif hasil_keywords == "Positif":
                    st.success(f"Hasil berdasarkan kata kunci: **{hasil_keywords}**")
                else:
                    st.warning("Sentimen tidak ditemukan dari kata kunci.")
            else:
                # Menambahkan warna untuk hasil sentimen berdasarkan model
                if hasil_model[0] == 0:
                    st.error(f"Sentimen Komentar: **Negatif**")
                elif hasil_model[0] == 1:
                    st.info(f"Sentimen Komentar: **Netral**")
                elif hasil_model[0] == 2:
                    st.success(f"Sentimen Komentar: **Positif**")
                else:
                    st.error("Prediksi tidak valid.")

        except ValueError as e:
            st.warning(f"Terjadi kesalahan pada transformasi input: {e}")