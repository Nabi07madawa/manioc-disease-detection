import os
import gradio as gr
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np


class MultimodalModel(nn.Module):
    def __init__(self, num_env_features=5, num_classes=5):
        super(MultimodalModel, self).__init__()
        resnet = models.resnet50(weights=None)
        self.image_branch = nn.Sequential(*list(resnet.children())[:-1])
        self.image_fc = nn.Sequential(
            nn.Linear(2048, 256), nn.ReLU(), nn.Dropout(0.3)
        )
        self.env_branch = nn.Sequential(
            nn.Linear(num_env_features, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU()
        )
        self.fusion = nn.Sequential(
            nn.Linear(256 + 32, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, image, env_data):
        img_features = self.image_branch(image)
        img_features = img_features.view(img_features.size(0), -1)
        img_features = self.image_fc(img_features)
        env_features = self.env_branch(env_data)
        combined = torch.cat([img_features, env_features], dim=1)
        return self.fusion(combined)


MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "best_multimodal_v2.pth")
MAX_IMAGE_SIZE = 10 * 1024 * 1024

device = torch.device("cpu")
model = MultimodalModel().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

maladies = {
    0: ("Bacteriose (CBB)", "Feuilles qui fletrissent et meurent. Traitement : eliminer les plantes infectees."),
    1: ("Striure brune (CBSD)", "Racines qui pourrissent. Traitement : utiliser des varietes resistantes."),
    2: ("Marbrure verte (CGM)", "Feuilles deformees. Traitement : controle des insectes vecteurs."),
    3: ("Mosaique (CMD)", "Taches jaune-vert sur les feuilles. Traitement : replanter avec des boutures saines."),
    4: ("Sain", "La plante est en bonne sante ! Continuez les bonnes pratiques agricoles.")
}


def predire_maladie(image, temperature, precipitation, humidite, vent, evapotranspiration):
    if image is None:
        return "Veuillez uploader une image"

    if not isinstance(image, np.ndarray):
        return "Format d'image non valide"

    if image.size > MAX_IMAGE_SIZE:
        return "Image trop volumineuse (max 10 Mo)"

    if image.ndim not in (2, 3):
        return "Format d'image non supporte"

    temperature = float(np.clip(temperature, 15, 40))
    precipitation = float(np.clip(precipitation, 0, 15))
    humidite = float(np.clip(humidite, 30, 100))
    vent = float(np.clip(vent, 0, 30))
    evapotranspiration = float(np.clip(evapotranspiration, 0, 10))

    img = Image.fromarray(image).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    env_data = torch.tensor(
        [[temperature, precipitation, humidite, vent, evapotranspiration]],
        dtype=torch.float32
    ).to(device)

    with torch.no_grad():
        outputs = model(img_tensor, env_data)
        probas = torch.softmax(outputs, dim=1)[0]
        pred = torch.argmax(probas).item()
        confiance = probas[pred].item() * 100

    nom_maladie, description = maladies[pred]
    resultat = f"Diagnostic : {nom_maladie}\n"
    resultat += f"Confiance : {confiance:.1f}%\n\n"
    resultat += f"{description}\n\n"
    resultat += "--- Probabilites par classe ---\n"
    for i in range(5):
        barre = "#" * int(probas[i].item() * 30)
        resultat += f"{maladies[i][0]:25s} : {probas[i].item()*100:5.1f}% {barre}\n"
    return resultat


demo = gr.Interface(
    fn=predire_maladie,
    inputs=[
        gr.Image(label="Photo de feuille de manioc"),
        gr.Slider(15, 40, value=26, step=0.5, label="Temperature (C)"),
        gr.Slider(0, 15, value=2.5, step=0.1, label="Precipitations (mm/jour)"),
        gr.Slider(30, 100, value=75, step=1, label="Humidite (%)"),
        gr.Slider(0, 30, value=10, step=0.5, label="Vitesse du vent (km/h)"),
        gr.Slider(0, 10, value=4, step=0.1, label="Evapotranspiration (mm/jour)")
    ],
    outputs=gr.Textbox(label="Resultat du diagnostic", lines=12),
    title="IA de Detection des Maladies du Manioc",
    description="Uploadez une photo de feuille de manioc et ajustez les conditions environnementales pour obtenir un diagnostic.",
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=10000)
