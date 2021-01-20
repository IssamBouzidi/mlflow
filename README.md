# MLFlow  
  
## introduction  
Utliser MLFlow pour le monitoring des resultats du modèle CNN de classification des images  
  
## Requirements  
Installer requirements.txt pour faire executer ce cide  
  
## Execution  
Il faut lancer le fichier main.py qui va entrainer le modèle CNN et faire le traçage de l'entrainement du modèle  
l'éxecution du fichier se fait par ligne de commande  

```
python main.py --filters=32 --epochs=20 --kernel_size=3  
```

La ligne de commandes peut prendre plisueurs arguments, ci-dessous la liste:  
```
parser = argparse.ArgumentParser()
parser.add_argument("--filters", help="Number of Filters", action='store', nargs='?', default=16,
                    type=int)
parser.add_argument("--hidden_layers", help="Number of Hidden Layers", action='store', nargs='?', default=1,
                    type=int)
parser.add_argument("--output", help="Output from First & Hidden Layers", action='store', nargs='?', default=2,
                    type=int)
parser.add_argument("--epochs", help="Number of epochs for training", nargs='?', action='store', default=20,
                    type=int)
parser.add_argument("--kernel_size", help="Number of epochs for training", nargs='?', action='store', default=3,
                    type=int)
parser.add_argument("--loss", help="Loss Function for the Gradients", nargs='?', action='store',
                    default='categorical_crossentropy', type=str)
parser.add_argument("--optimizer", help="Optimizer", nargs='?', action='store', default='adam', type=str)
parser.add_argument("--load_model_path", help="Load model path", nargs='?', action='store', default='/tmp', type=str)
parser.add_argument("--my_review", help="Type in your review", nargs='?', action='store', default='this film was horrible, bad acting, even worse direction', type=str)
parser.add_argument("--verbose", help="Verbose output", nargs='?', action='store', default=0, type=int)
parser.add_argument("--run_uuid", help="Specify the MLflow Run ID", nargs='?', action='store', default=None, type=str)
parser.add_argument("--tracking_server", help="Specify the MLflow Tracking Server", nargs='?', action='store', default=None, type=str)
parser.add_argument("--experiment_name", help="Name of the MLflow Experiment for the runs", nargs='?', action='store', default='Keras_CNN_Classifier', type=str)
```

## Résultat  
Une fois le mlflow excecuté, lancer le daschbord mlflow sur localhost:5000 pour visualier les resultat du traçage.  
  
### Captures écran résultat  
#### Informations générales  
![Informations Generales](results/screenshots/1.png)  

#### Parameters  
![Parameters](results/screenshots/2.png)  

#### Metrics  
![Metrics](results/screenshots/3.png)  

#### Exemple graph  
![Graph](results/screenshots/4.png)  
