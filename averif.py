from flask import Flask, request, render_template, jsonify
import joblib
import pickle

app = Flask(__name__)

# Charger les modèles et données
model = pickle.load(open("model.pkl", "rb"))
vectorizer = joblib.load(open("tfidf_vectorizer.pkl", "rb"))
#similarity_matrix = pickle.load(open("similarity_df1.pkl", "rb"))  # Exemple de matrice de similarité pré-calculée
#naemas_labels = pickle.load(open("naemas_labels.pkl", "rb"))  # Les étiquettes du modèle
#labels = pickle.load(open("labels.pkl", "rb"))
#corpus = pickle.load(open("corpus_tfidf.pkl", "rb"))  # Corpus utilisé pour la classification

@app.route('/')
def home():
    return render_template('averif_class.html')

@app.route('/classifier', methods=['POST'])
def classifier():
    try:
        # Récupération des données JSON
        data = request.json
        designation = data.get('DESIGNATION_PRECISE_ACTIVITE', '').strip()

        if not designation:
            return jsonify({"error": "Le champ DESIGNATION_PRECISE_ACTIVITE est requis."}), 400

        # Prédire le NAEMAS
        input_vector = vectorizer.transform([designation])
        predicted_naemas = model.predict(input_vector)[0]
        predicted_naemas_3 = predicted_naemas[:3]

        # Réponse JSON simplifiée
        return jsonify({
            "nouvelle_activite": designation,
            "classification": {
                "NAEMAS": predicted_naemas,
                "NAEMAS_3": predicted_naemas_3
            }
        })
    except Exception as e:
        return jsonify({"error": f"Erreur lors de la classification : {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
