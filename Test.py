import unittest
from ML_flow import load_data
import numpy as np


class PruebaML(unittest.TestCase):
    def test_model(self):
        model = load_data()
        predictions, tuned_dt,msj, exp_clf101 = load_data(ingenieria=True, modelo = 2)
        u1,u2 = load_data.precision(tuned_dt, exp_clf101)

        if msj == "Proceso Exitoso":
            self.assertLessEqual(np.abs(u1-u2),10,print("No hay Underfitting ni Overfitting"))
            a = "Modelo entrenado correctamente"
            return {'Procces':a}
        else:
            a = "Hay un error en el modelo que no permite completarlo de forma correcta"
            return {'Procces':a}

pb = PruebaML()
print(pb.test_model())
if __name__ == "__main__":
 unittest.main()