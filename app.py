from flask import Flask, request, jsonify
import face_recognition

app = Flask(__name__)

@app.route('/compare-images', methods=['POST'])
def compare_images():
    # Verifica se as duas imagens foram enviadas na requisição
    if 'cnh' not in request.files or 'camera' not in request.files:
        return jsonify({'error': 'Ambas as imagens (CNH e câmera) devem ser enviadas.'}), 400

    # Carrega as imagens do formulário
    cnh_image = face_recognition.load_image_file(request.files['cnh'])
    camera_image = face_recognition.load_image_file(request.files['camera'])

    # Encontra os rostos nas imagens
    cnh_face_locations = face_recognition.face_locations(cnh_image)
    camera_face_locations = face_recognition.face_locations(camera_image)

    # Verifica se foram encontrados exatamente um rosto em cada imagem
    if len(cnh_face_locations) != 1 or len(camera_face_locations) != 1:
        return jsonify({'error': 'Deve haver exatamente um rosto em cada imagem.'}), 400

    # Extrai os descritores faciais das imagens
    cnh_face_encoding = face_recognition.face_encodings(cnh_image, cnh_face_locations)[0]
    camera_face_encoding = face_recognition.face_encodings(camera_image, camera_face_locations)[0]

    # Compara os descritores faciais para verificar a similaridade
    face_distance = face_recognition.face_distance([cnh_face_encoding], camera_face_encoding)[0]

    # Define um limite de similaridade
    threshold = 0.6

    # Verifica se as imagens são da mesma pessoa com base na similaridade
    if face_distance <= threshold:
        return jsonify({'result': 'As imagens são da mesma pessoa.'}), 200
    else:
        return jsonify({'result': 'As imagens são de pessoas diferentes.'}), 200

if __name__ == '__main__':
    app.run(debug=True)
