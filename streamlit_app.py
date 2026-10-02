import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import os


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


@st.cache_resource
def charger_modele():
    device = torch.device("cpu")
    model = MultimodalModel().to(device)
    model_path = os.path.join(os.path.dirname(__file__), "models", "best_multimodal_v2.pth")
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    return model, device


model, device = charger_modele()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

maladies = {
    0: ("Bacteriose (CBB)", "Feuilles qui fletrissent et meurent.\n\n**Traitement :** Eliminer les plantes infectees."),
    1: ("Striure brune (CBSD)", "Racines qui pourrissent.\n\n**Traitement :** Utiliser des varietes resistantes."),
    2: ("Marbrure verte (CGM)", "Feuilles deformees.\n\n**Traitement :** Controle des insectes vecteurs."),
    3: ("Mosaique (CMD)", "Taches jaune-vert sur les feuilles.\n\n**Traitement :** Replanter avec des boutures saines."),
    4: ("Sain", "La plante est en bonne sante !\n\nContinuez les bonnes pratiques agricoles.")
}

st.title("IA de Detection des Maladies du Manioc")
st.write("Uploadez une photo de feuille de manioc et ajustez les conditions environnementales pour obtenir un diagnostic.")

col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("Photo de feuille de manioc", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Image uploadee", use_container_width=True)

with col2:
    st.subheader("Conditions environnementales")
    temperature = st.slider("Temperature (C)", 15.0, 40.0, 26.0, 0.5)
    precipitation = st.slider("Precipitations (mm/jour)", 0.0, 15.0, 2.5, 0.1)
    humidite = st.slider("Humidite (%)", 30, 100, 75, 1)
    vent = st.slider("Vitesse du vent (km/h)", 0.0, 30.0, 10.0, 0.5)
    evapotranspiration = st.slider("Evapotranspiration (mm/jour)", 0.0, 10.0, 4.0, 0.1)

if uploaded_file is not None:
    if st.button("Lancer le diagnostic", type="primary"):
        with st.spinner("Analyse en cours..."):
            img_tensor = transform(image).unsqueeze(0).to(device)
            env_data = torch.tensor(
                [[temperature, precipitation, float(humidite), vent, evapotranspiration]],
                dtype=torch.float32
            ).to(device)

            with torch.no_grad():
                outputs = model(img_tensor, env_data)
                probas = torch.softmax(outputs, dim=1)[0]
                pred = torch.argmax(probas).item()
                confiance = probas[pred].item() * 100

            nom_maladie, description = maladies[pred]

            if pred == 4:
                st.success(f"**{nom_maladie}** — Confiance : {confiance:.1f}%")
            else:
                st.error(f"**{nom_maladie}** — Confiance : {confiance:.1f}%")

            st.write(description)

            st.subheader("Probabilites par classe")
            for i in range(5):
                prob = probas[i].item() * 100
                st.write(f"**{maladies[i][0]}**")
                st.progress(probas[i].item())
                st.caption(f"{prob:.1f}%")
