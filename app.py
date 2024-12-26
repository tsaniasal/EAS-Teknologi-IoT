from flask import Flask, render_template, request
import numpy as np
from skfuzzy.control import Antecedent, Consequent, Rule, ControlSystem, ControlSystemSimulation
from skfuzzy.membership import trapmf, trimf
import tensorflow as tf
from train_data import train_data, train_labels

app = Flask(__name__)

def fuzzy_logic(suhu_input, tegangan_input, arus_input, ultrasonik_input, radiasi_input):
    # Definisikan variabel fuzzy untuk 5 input sensor
    suhu = Antecedent(np.arange(0, 43, 1), 'suhu')  # Rentang suhu 0-42
    tegangan = Antecedent(np.arange(0, 11, 1), 'tegangan')  # Rentang tegangan 0-10
    arus = Antecedent(np.arange(0, 21, 1), 'arus')  # Rentang arus 0-20
    ultrasonik = Antecedent(np.arange(0, 51, 1), 'ultrasonik')  # Rentang ultrasonik 0-50
    radiasi = Antecedent(np.arange(0, 1002, 1), 'radiasi')  # Rentang radiasi 0-1001

    # Output untuk 5 aktuator
    motor_speed = Consequent(np.arange(0, 11, 1), 'motor_speed')  # Rentang motor_speed 0-10
    linear_actuator = Consequent(np.arange(0, 11, 1), 'linear_actuator')  # Rentang linear_actuator 0-10
    pompa_air = Consequent(np.arange(0, 11, 1), 'pompa_air')  # Rentang pompa_air 0-10
    nozzle_sprayer = Consequent (np.arange(0, 11, 1), 'nozzle_sprayer')  # Rentang nozzle_sprayer 0-10
    rel_geser = Consequent (np.arange(0, 31, 1), 'rel_geser')  # Rentang rel_geser 0-30

    # Fungsi keanggotaan untuk input (low, medium, high)
    # Keanggotaan untuk suhu
    suhu['low'] = trimf(suhu.universe, [0, 0, 14])  # Low antara 0 dan 14
    suhu['medium'] = trimf(suhu.universe, [10, 21, 32])  # Medium antara 10 dan 32
    suhu['high'] = trimf(suhu.universe, [28, 43, 43])  # High antara 28 dan 43

    # Keanggotaan untuk tegangan
    tegangan['low'] = trimf(tegangan.universe, [0, 0, 3])  # Low antara 0 dan 34
    tegangan['medium'] = trimf(tegangan.universe, [2, 5, 8])  # Medium antara 30 dan 70
    tegangan['high'] = trimf(tegangan.universe, [7, 10, 10])  # High antara 60 dan 101

    # Keanggotaan untuk arus
    arus['low'] = trimf(arus.universe, [0, 0, 7])  # Low antara 0 dan 3
    arus['medium'] = trimf(arus.universe, [5, 10, 15])  # Medium antara 2 dan 8
    arus['high'] = trimf(arus.universe, [13, 20, 20])  # High antara 7 dan 11

    # Keanggotaan untuk ultrasonik
    ultrasonik['low'] = trimf(ultrasonik.universe, [0, 0, 17])  # Low antara 0 dan 17
    ultrasonik['medium'] = trimf(ultrasonik.universe, [10, 25, 40])  # Medium antara 10 dan 40
    ultrasonik['high'] = trimf(ultrasonik.universe, [30, 51, 51])  # High antara 30 dan 51

    # Keanggotaan untuk radiasi
    radiasi['low'] = trimf(radiasi.universe, [0, 0, 300])  # Low antara 0 dan 300
    radiasi['medium'] = trimf(radiasi.universe, [200, 400, 600])  # Medium antara 200 dan 600
    radiasi['high'] = trimf(radiasi.universe, [500, 1001, 1001])  # High antara 500 dan 1001


    # Fungsi keanggotaan untuk output (motor_speed, linear_actuator, pompa_air, nozzle_sprayer,rel_geser )
    motor_speed['low'] = trimf(motor_speed.universe, [0, 0, 3])  # Low antara 0 dan 3
    motor_speed['medium'] = trimf(motor_speed.universe, [2, 5, 8])  # Medium antara 2 dan 8
    motor_speed['high'] = trimf(motor_speed.universe, [7, 10, 10])  # High antara 7 dan 10

    linear_actuator['low'] = trimf(linear_actuator.universe, [0, 0, 3])  # Low antara 0 dan 3
    linear_actuator['medium'] = trimf(linear_actuator.universe, [ 2, 5, 8])  # Medium antara 2 dan 8
    linear_actuator['high'] = trimf(linear_actuator.universe, [7, 10, 10])  # High antara 7 dan 10

    pompa_air['low'] = trimf(pompa_air.universe, [0, 0, 3])  # Low antara 0 dan 3
    pompa_air['medium'] = trimf(pompa_air.universe, [2, 5, 8])  # Medium antara 2 dan 8
    pompa_air['high'] = trimf(pompa_air.universe, [7, 10, 10])  # High antara 7 dan 10

    nozzle_sprayer['low'] = trimf(nozzle_sprayer.universe, [0, 0, 3])  # Low antara 0 dan 3
    nozzle_sprayer ['medium'] = trimf(nozzle_sprayer.universe, [2, 5, 8])  # Medium antara 2 dan 8
    nozzle_sprayer ['high'] = trimf(nozzle_sprayer.universe, [7, 10, 10])  # High antara 7 dan 10

    rel_geser = Antecedent(np.arange(0, 31, 1), 'rel_geser')  # Rentang rel geser 0-30
    rel_geser['low'] = trimf(rel_geser.universe, [0, 0, 10])  # Low antara 0 dan 10
    rel_geser['medium'] = trimf(rel_geser.universe, [5, 15, 25])  # Medium antara 5 dan 25
    rel_geser['high'] = trimf(rel_geser.universe, [20, 30, 30])  # High antara 20 dan 30


    # Aturan fuzzy untuk output motor, linear actuator, dan pompa air
    rule1 = Rule(arus['low'] , motor_speed['low'])
    rule2 = Rule(arus['medium'], motor_speed['medium'])
    rule3 = Rule(arus['high'], motor_speed['high'])

    rule4 = Rule(suhu['low'], linear_actuator['low'])
    rule5 = Rule(suhu['medium'], linear_actuator['medium'])
    rule6 = Rule(suhu['high'], linear_actuator['high'])

    rule7 = Rule(tegangan['low'], pompa_air['low'])
    rule8 = Rule(tegangan['medium'], pompa_air['medium'])
    rule9 = Rule(tegangan['high'], pompa_air['high'])

    rule10 = Rule(ultrasonik['low'], rel_geser['low'])
    rule11 = Rule(ultrasonik ['medium'], rel_geser ['medium'])
    rule12 = Rule(ultrasonik ['high'], rel_geser ['high'])

    rule13 = Rule(radiasi['low'], nozzle_sprayer ['low'])
    rule14 = Rule(radiasi ['medium'], nozzle_sprayer ['medium'])
    rule15 = Rule(radiasi ['high'], nozzle_sprayer ['high'])


    # Sistem kontrol fuzzy
    motor_ctrl = ControlSystem([rule1, rule2, rule3])
    linear_ctrl = ControlSystem([rule4, rule5, rule6])
    pompa_ctrl = ControlSystem([rule7, rule8, rule9])
    rel_ctrl = ControlSystem([rule10, rule11, rule12])
    nozzle_ctrl = ControlSystem([rule13, rule14, rule15])

    motor_simulation = ControlSystemSimulation(motor_ctrl)
    linear_simulation = ControlSystemSimulation(linear_ctrl)
    pompa_simulation = ControlSystemSimulation(pompa_ctrl)
    rel_simulation = ControlSystemSimulation(rel_ctrl)
    nozzle_simulation = ControlSystemSimulation(nozzle_ctrl)

    # Input untuk sensor
    try:
        # debug motor_speed
        print("MOTOR Speed")
        motor_simulation.input['arus'] = arus_input
        print(motor_simulation.input)

        # debug linear_actuator
        print("LINEAR Actuator")
        linear_simulation.input['suhu'] = suhu_input
        print(linear_simulation.input)

        # debug pompa_air
        print("POMPA Air")
        pompa_simulation.input['tegangan'] = tegangan_input
        print(pompa_simulation.input)

        # debug rel_geser
        print("REL Geser")
        rel_simulation.input['ultrasonik'] = ultrasonik_input
        print(rel_simulation.input)

        # debug nozzle_sprayer
        print("NOZZLE_Sprayer")
        nozzle_simulation.input['radiasi'] = radiasi_input
        print(nozzle_simulation.input)


        # Menjalankan simulasi
        motor_simulation.compute()
        linear_simulation.compute()
        pompa_simulation.compute()
        rel_simulation.compute()
        nozzle_simulation.compute()
        print("DONE COMPUTE")
        print(motor_simulation.output)
        print(linear_simulation.output)
        print(pompa_simulation.output)
        print(rel_simulation.output)
        print(nozzle_simulation.output)

        # Output yang dihitung
        return motor_simulation.output['motor_speed'], linear_simulation.output['linear_actuator'], pompa_simulation.output['pompa_air'], rel_simulation.output['rel_geser'], nozzle_simulation.output ['nozzle_sprayer']
    except Exception as e:
        print(f"Error: {e}")  # Cetak error untuk debugging
        return 0, 0, 0, 0, 0

# Neural Network untuk prediksi output (contoh sederhana)
def neural_network_predict(suhu, tegangan, arus, ultrasonik, radiasi, train_data, train_labels, epochs=100):
    # Model Neural Network untuk prediksi output
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_dim=5),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(5)  # Output 5 Aktuator
    ])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

    # Training model
    model.fit(train_data, train_labels, epochs=epochs, verbose=1)

    # Input untuk prediksi
    input_data = np.array([[suhu, tegangan, arus, ultrasonik, radiasi]])

    # Prediksi output
    output = model.predict(input_data)
    return output[0][0], output[0][1], output[0][2], output[0][3], output[0][4]

# Halaman utama
@app.route('/')
def index():
    return render_template('index.html',  motor_speed=0, linear_actuator=0, pompa_air=0, rel_geser=0, nozzle_sprayer=0,
                               nn_motor=0, nn_linear=0, nn_pompa=0, nn_rel=0, nn_nozzle=0)

# Proses data dari frontend
@app.route('/process', methods=['POST'])
def process():
    if request.method == 'POST':
        suhu = float(request.form['suhu'])
        tegangan = float(request.form['tegangan'])
        arus = float(request.form['arus'])
        ultrasonik = float(request.form['ultrasonik'])
        radiasi = float(request.form['radiasi'])

        print("====================================")
        print("Data Input")
        print("Suhu:", suhu, "\nTegangan:", tegangan, "\nArus:", arus, "\nUltrasonik:", ultrasonik, "\nRadiasi:", radiasi)
        print("====================================")

        # Proses dengan Fuzzy Logic
        print("Fuzzy Logic")
        motor_output, linear_output, pompa_output, rel_output, nozzle_output = fuzzy_logic(suhu, tegangan, arus, ultrasonik, radiasi)

        # Proses dengan Neural Network
        print("Neural Network")

        nn_motor, nn_linear, nn_pompa, nn_rel, nn_nozzle = neural_network_predict(suhu, tegangan, arus, ultrasonik, radiasi, train_data, train_labels)

        # debug print
        print("====================================")
        print(f"Motor Speed: {motor_output}")
        print(f"Linear Actuator: {linear_output}")
        print(f"Pompa Air: {pompa_output}")
        print(f"Rel Geser: {rel_output}")
        print(f"Nozzle Sprayer: {nozzle_output}")
        print(f"NN Motor: {nn_motor}")
        print(f"NN Linear: {nn_linear}")
        print(f"NN Pompa: {nn_pompa}")
        print(f"NN Rel Geser: {nn_rel}")
        print(f"NN Nozzle Sprayer: {nn_nozzle}")

        # Mengirim hasil ke frontend
        return render_template('index.html', motor_speed=motor_output, linear_actuator=linear_output, pompa_air=pompa_output, rel=rel_output, nozzle=nozzle_output,
                               nn_motor=nn_motor, nn_linear=nn_linear, nn_pompa=nn_pompa, nn_rel=nn_rel, nn_nozzle=nn_nozzle)

if __name__ == '__main__':
    app.run(debug=True)
