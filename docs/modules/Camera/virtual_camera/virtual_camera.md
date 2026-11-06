# Virtual Camera Module

## Description
Le module **Virtual Camera** permet de diffuser des flux d'images (`IOTypes.FRAME`) générés ou traités dans `Pipeline_creator` vers une caméra virtuelle reconnue par le système d'exploitation (Windows, macOS, Linux).

Cette caméra virtuelle peut être sélectionnée comme webcam d'entrée dans n'importe quelle application tierce (ex: **OBS Studio**, **Microsoft Teams**, **Zoom**, **Discord**, **Google Meet**, **navigateurs Web**, etc.).

---

## Fonctionnalités
- **Support multi-backends** :
  - `obs` : Pilote virtuel DirectShow natif d'OBS Studio.
  - `unitycapture` : Pilote Unity Capture.
  - `akvcam` : AkVirtualCamera.
  - `v4l2loopback` : Linux v4l2.
  - `auto` : Détection automatique du meilleur backend disponible.
- **Conversion automatique des formats** :
  - Supporte les images Niveaux de gris (1 canal), BGR/RGB (3 canaux) et BGRA/RGBA (4 canaux).
  - Redimensionnement automatique si la taille des frames d'entrée diffère de la résolution configurée.
- **Option Auto-fit** :
  - Détecte et adapte automatiquement la résolution de la caméra virtuelle à la première frame reçue.
- **Télémétrie en direct** :
  - Compteur de frames envoyées.
  - Calcul et affichage du débit d'images effectif (FPS).
  - État du périphérique et gestion des erreurs en temps réel.
- **Passthrough de sortie** :
  - Sortie `Color Frame (BGR)` pour continuer le chaînage dans le pipeline.
  - Sortie `Status` pour surveiller l'état du module (dictionnaire de statut).

---

## Prérequis

1. **Bibliothèque Python** :
   ```bash
   pip install pyvirtualcam
   ```

2. **Pilote de Caméra Virtuelle** (sur Windows) :
   - **OBS Studio** (recommandé) : Installer OBS Studio installe automatiquement le driver de caméra virtuelle DirectShow.
   - ou **Unity Capture** : Télécharger et installer le pilote DirectShow Unity Capture.

---

## Entrées / Sorties

### Entrées
- `Color Frame (BGR)` (`IOTypes.FRAME`) : Flux d'images numpy (`np.ndarray` ou tuple `(np.ndarray, str)`).

### Sorties
- `Color Frame (BGR)` (`IOTypes.FRAME`) : Passthrough du flux d'images reçu.
- `Status` (`IOTypes.STATUS_DICT`) : Dictionnaire contenant `{is_running, device, fps, frames_sent}`.
