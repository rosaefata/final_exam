import category_encoders as ce
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

def get_oiliness_skin_type(oiliness_ans):
    #load data
    data = pd.DataFrame(oiliness_ans, columns=['1', '2', '3', '4', '6', '7', '8', '11'])
    print(data)

    #encode data
    dictionary = [
        {'col': '1',
         'mapping': {'Sangat kasar, berkelupas, atau pucat': 1, 'Ketat': 2, 'Terhidrasi dengan baik tanpa kilapan': 3,
                     'Mengkilap': 4}},
        {'col': '2',
         'mapping': {'Tidak pernah, atau Anda tidak pernah menyadari kilauan': 1, 'Kadang-kadang': 2, 'Sering': 3,
                     'Selalu': 4}},
        {'col': '3', 'mapping': {'Terkelupas atau berlapis kerutan': 1, 'Halus': 2, 'Berkilau': 4,
                                 'Bergaris-garis dan mengkilap': 5, 'Saya tidak mengunakan foundation': 3}},
        {'col': '4', 'mapping': {'Terasa sangat kering atau pecah-pecah': 1, 'Terasa kencang': 2, 'Tidak tahu': 3,
                                 'Terasa biasa': 4,
                                 'Terlihat berkilau, atau saya tidak pernah merasa membutuhkan pelembab': 5}},

        {'col': '6', 'mapping': {'Kering': 1, 'Normal': 2, 'Kombinasi': 3, 'Berminyak': 4}},
        {'col': '7', 'mapping': {'Terasa kering atau pecah-pecah': 1, 'Terasa sedikit kering tapi tidak pecah-pecah': 2,
                                 'Terasa biasa saja': 4, 'Terasa berminyak': 5,
                                 'Saya tidak menggunakan sabun atau pembersih berbusa lainnya. (Jika ini karena sabun berbusa membuat kulit Anda kering, pilih pilihan pertama)': 3}},
        {'col': '8', 'mapping': {'Selalu': 1, 'Kadang-kadang': 2, 'Jarang': 3, 'Tidak pernah': 4}},
        {'col': '11', 'mapping': {'Sangat kasar, pecah-pecah, atau pucat': 1, 'Halus': 2, 'Sedikit mengkilap': 3,
                                  'Mengkilap dan licin, atau saya tidak menggunakan pelembab': 4}}
    ]

    data_encoder = ce.OrdinalEncoder(cols=('1', '2', '3', '4', '6', '7', '8', '11'), mapping=dictionary)
    x_encoded = data_encoder.fit_transform(data)
    print(x_encoded)
    #predict
    model = XGBClassifier()
    model.load_model("oiliness.json")
    prediction_result = model.predict(x_encoded)
    proba = model.predict_proba(x_encoded)

    print('Oiliness:', proba)

    if prediction_result[0] == 0:
        return 'D', proba[0][0]
    else:
        return 'O', proba[0][1]

def get_tightness_skin_type(tightness_ans):
    # load data
    data = pd.DataFrame(tightness_ans, columns=['1', '2', '3', '4', '5', '7', '8', '9','11','12','13','14','15'])
    print(data)
    dictionary = [
        {'col': '1', 'mapping': {'Tidak': 1,
                                 'Hanya ketika saya menggerakkan wajah, seperti tersenyum, mengerutkan kening, atau mengangkat alis': 2,
                                 'Ya, sedikit saat menggerakkan wajah dan sedikit saat diam tanpa gerakan': 3,
                                 'Kerutan muncul bahkan jika saya tidak tersenyum, mengerutkan kening, atau mengangkat alis': 4}},
        {'col': '2', 'mapping': {'Lima hingga sepuluh tahun lebih muda dari usianya': 1, 'Sesuai usianya': 2,
                                 'Tak dapat diterapkan. Saya diadopsi atau saya tidak ingat': 3,
                                 'Lima tahun lebih tua dari usianya': 4,
                                 'Lebih dari lima tahun lebih tua dari usianya': 5}},
        {'col': '3', 'mapping': {'Lima hingga sepuluh tahun lebih muda dari usianya': 1, 'Sesuai usianya': 2,
                                 'Lima tahun lebih tua dari usianya': 4,
                                 'Lebih dari lima tahun lebih tua dari usianya': 5,
                                 'Tak dapat diterapkan. Saya diadopsi atau saya tidak ingat': 3}},
        {'col': '4', 'mapping': {'Tidak pernah': 1, 'Satu sampai lima tahun': 2, 'Lima sampai sepuluh tahun': 3,
                                 'Lebih dari sepuluh tahun': 4}},
        {'col': '5', 'mapping': {'Tidak pernah': 1, 'Satu sampai lima tahun': 2, 'Lima sampai sepuluh tahun': 3,
                                 'Lebih dari sepuluh tahun': 4}},

        {'col': '7', 'mapping': {'Satu sampai lima tahun lebih muda dari usiamu': 1, 'Sesuai usia': 2,
                                 'Lima tahun lebih tua dari usiamu': 3,
                                 'Lebih dari lima tahun lebih tua dari usiamu': 4}},
        {'col': '8', 'mapping': {'Tidak pernah': 1, 'Sekali sebulan': 2, 'Sekali seminggu': 3, 'Sehari-hari': 4}},
        {'col': '9',
         'mapping': {'Tidak pernah': 1, 'Satu sampai lima kali': 2, 'Lima sampai sepuluh kali': 3, 'Berkali-kali': 4}},

        {'col': '11', 'mapping': {'Udaranya segar dan bersih.': 1, 'Udara sangat tercemar.': 2}},
        {'col': '12',
         'mapping': {'Bertahun-tahun': 1, 'Kadang-kadang': 2, 'Sekali untuk jerawat ketika saya masih muda': 3,
                     'Tidak pernah': 4}},
        {'col': '13', 'mapping': {'Di setiap makan': 1, 'Sekali sehari': 2, 'Kadang-kadang': 3, 'Tidak pernah': 4}},
        {'col': '14', 'mapping': {'75-100 persen': 1, '25-75 persen': 2, '10-25 persen': 3, '0-10 persen': 4}},
        {'col': '15', 'mapping': {'Gelap': 1, 'Sedang': 2, 'Putih': 3, 'Sangat putih': 4}},

    ]

    data_encoder = ce.OrdinalEncoder(cols=('1', '2', '3', '4', '5', '7', '8', '9','11','12','13','14','15'), mapping=dictionary)
    x_encoded = data_encoder.fit_transform(data)
    print(x_encoded)
    # predict
    model = XGBClassifier()
    model.load_model("tightness.json")
    prediction_result = model.predict(x_encoded)
    proba = model.predict_proba(x_encoded)

    print('Tightness:', proba)

    if prediction_result[0] == 0:
        return 'T', proba[0][0]
    else:
        return 'W', proba[0][1]

def get_sensitivity_skin_type(sensitivity_ans):
    # load data
    data = pd.DataFrame(sensitivity_ans, columns=['1', '2', '3', '4', '5', '7', '8', '9', '12', '14', '15','17','18'])
    print(data)
    dictionary = [
        {'col': '1', 'mapping': {'Tidak pernah': 1, 'Minimal sebulan sekali': 2}},
        {'col': '2', 'mapping': {'Tidak pernah': 1, 'Sering': 2}},
        {'col': '3', 'mapping': {'Tidak': 1, 'Ya': 2}},
        {'col': '4', 'mapping': {'Tidak pernah': 1, 'Sering': 2}},
        {'col': '5', 'mapping': {'Tidak pernah': 1, 'Sering': 2}},
        {'col': '7', 'mapping': {'Tidak pernah': 1, 'Sering': 2}},
        {'col': '8', 'mapping': {'Tidak pernah': 1, 'Sering': 2}},
        {'col': '9', 'mapping': {'Ya': 1, 'Tidak, kulit saya gatal, memerah, atau pecah-pecah.': 2}},
        {'col': '12', 'mapping': {'Tidak pernah': 1, 'Sering': 2}},
        {'col': '14', 'mapping': {'Tidak pernah': 1, 'Sering': 2}},
        {'col': '15', 'mapping': {'Tidak ada': 1, 'Sedikit (satu sampai tiga di seluruh wajah, termasuk hidung)': 2,
                                  'Beberapa (empat hingga enam di seluruh wajah, termasuk hidung)': 3,
                                  'Banyak (lebih dari tujuh di seluruh wajah, termasuk hidung)': 4}},
        {'col': '17', 'mapping': {'Tidak pernah': 1, 'Kadang-kadang': 2, 'Sering': 3, 'Selalu': 4}},
        {'col': '18', 'mapping': {'Tidak pernah': 1, 'Sering': 2}}
    ]
    data_encoder = ce.OrdinalEncoder(cols=('1', '2', '3', '4', '5', '7', '8', '9', '12', '14', '15','17','18'),
                                     mapping=dictionary)
    x_encoded = data_encoder.fit_transform(data)
    print(x_encoded)
    # predict
    model = XGBClassifier()
    model.load_model("sensitivity.json")
    prediction_result = model.predict(x_encoded)
    proba = model.predict_proba(x_encoded)

    print('Sensitivity:', proba)

    if prediction_result[0] == 0:
        return 'R', proba[0][0]
    else:
        return 'S', proba[0][1]

def get_pigmentation_skin_type(pigmentation_ans):

    data = pd.DataFrame(pigmentation_ans, columns=['1', '2', '3', '4', '5', '7', '11'])
    print(data)
    dictionary = [
        {'col': '1',
         'mapping': {'Tidak pernah atau saya tidak memperhatikan': 1, 'Kadang-kadang': 2, 'Sering': 3, 'Selalu': 4}},
        {'col': '2', 'mapping': {'Saya tidak mendapatkan bekas luka atau saya tidak memperhatikan': 1, 'Seminggu': 2,
                                 'Beberapa minggu': 3, 'Berbulan-bulan': 4}},
        {'col': '3', 'mapping': {'Tidak ada': 1, 'Satu': 2, 'Beberapa': 3, 'Banyak': 4}},
        {'col': '4',
         'mapping': {'Tidak': 1, 'Saya tidak yakin.': 2, 'Ya, sedikit terlihat.': 3, 'Ya, sangat mencolok.': 4}},
        {'col': '5', 'mapping': {'Saya tidak memiliki flek hitam': 1, 'Tidak yakin': 2, 'Sedikit lebih buruk': 3,
                                 'Jauh lebih buruk': 4}},
        {'col': '7', 'mapping': {'Tidak': 1, 'Ya, beberapa (satu sampai lima)': 2, 'Ya, ton (enam belas atau lebih)': 4,
                                 'Ya, banyak (enam sampai lima belas)': 3}},

        {'col': '11', 'mapping': {'Ya': 1, 'Tidak': 2}}
    ]
    data_encoder = ce.OrdinalEncoder(cols=('1', '2', '3', '4', '5', '7', '11'),
                                     mapping=dictionary)

    x_encoded = data_encoder.fit_transform(data)
    print(x_encoded)
    # predict
    model = XGBClassifier()
    model.load_model("pigmentation.json")
    prediction_result = model.predict(x_encoded)
    proba = model.predict_proba(x_encoded)

    print('Pigmentation:', proba)

    if prediction_result[0] == 0:
        return 'N', proba[0][0]
    else:
        return 'P', proba[0][1]


