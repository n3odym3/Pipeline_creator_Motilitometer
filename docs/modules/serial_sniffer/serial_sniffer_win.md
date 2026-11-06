# Serial Sniffer & Reverse-Engineering Workbench (com0com)

Le module **Serial Sniffer** est un outil générique d'interception, de journalisation, d'analyse et de rejeu de communications séries (USB/RS-232). Il implémente une architecture **Man-In-The-Middle (MITM)** via une paire de ports virtuels **com0com** pour s'intercaler en toute transparence entre n'importe quel logiciel propriétaire sur PC et un appareil physique.

Ce module est conçu pour la **rétro-ingénierie et l'analyse de protocoles séries propriétaires**, quel que soit le matériel (capteurs, instruments de mesure, automates, contrôleurs).

---

## 1. Principe de l'Architecture MITM avec com0com

En fonctionnement classique :
```
[Logiciel Propriétaire / App] <====== COM Réel ======> [Appareil Physique / Hardware]
```

En mode Sniffer avec com0com :
```
[Logiciel Propriétaire / App]
       │
       ▼ (se connecte à COM10)
[com0com : Paire Virtuelle COM10 <===========> COM11]
                                                 │
                                                 ▼ (se connecte à COM11)
                                      ┌──────────────────────────────┐
                                      │        Serial Sniffer        │
                                      │  - Transfert bidirectionnel  │
                                      │  - Découpage intelligent     │
                                      │  - Moniteur Hex + ASCII      │
                                      │  - Enregistreur / Replay     │
                                      └──────────────┬───────────────┘
                                                     │
                                                     ▼ (se connecte au COM Réel, ex: COM3)
                                         [Appareil Physique / Hardware]
```

Toutes les commandes émises par le logiciel (TX : `APP->DEV`) et toutes les réponses de l'appareil (RX : `DEV->APP`) sont interceptées en temps réel avec un horodatage haute précision (delta ms), affichées à l'écran, enregistrées dans un fichier de session, et peuvent être rejouées.

---

## 2. Configuration Initiale de com0com (Windows)

1. Téléchargez et installez **com0com** (Null-modem emulator).
2. Lancez l'utilitaire **Setup Command Prompt** ou le panneau graphique de com0com en mode Administrateur.
3. Créez une paire de ports virtuels liés, par exemple :
   - Port A : `COM10`
   - Port B : `COM11`
   *(Remarque : activez l'option d'émulation de débit si nécessaire et décochez "enable buffer overrun")*.
4. Vérifiez dans le Gestionnaire de Périphériques Windows que `COM10` et `COM11` apparaissent bien.

---

## 3. Guide de Mise en Route Générique

1. **Identifier le port de votre appareil physique** :
   - Branchez votre appareil en USB / Série.
   - Repérez le port COM assigné par Windows (ex: `COM3` ou `COM4`).
2. **Configurer le Serial Sniffer** :
   - Dans le module **Serial Sniffer** :
     - **Physical Device** : sélectionnez le port physique (ex: `COM3`).
     - **App Port (com0com)** : sélectionnez le second port de la paire virtuelle (ex: `COM11`).
     - **Baudrate** : réglez la vitesse de transmission (9600, 19200, 38400, 57600, 115200, etc.).
     - **Parity / Stop** : None (N) / 1 stop bit (8N1 standard) ou selon votre matériel.
     - **Framing** : choisissez le découpage adapté (`Auto (\r / \n / Silence)`, `CR (\r)`, `CRLF (\r\n)`, `LF (\n)`, `STX/ETX` ou `Silence Only`).
   - Cliquez sur **Start Sniffer / MITM Bridge** (le bouton passe au vert et le statut indique *Running*).
3. **Configurer votre logiciel de contrôle propriétaire** :
   - Ouvrez votre logiciel constructeur habituel.
   - Dans les paramètres de communication / connexion, sélectionnez **COM10** (le premier port de la paire virtuelle).
   - Cliquez sur **Connecter**.
4. **Observer & Enregistrer** :
   - Vous verrez immédiatement défiler les trames d'initialisation, les requêtes et les réponses dans l'onglet **Live Traffic Monitor**.

---

## 4. Fonctionnalités Détaillées

### 4.1. Moniteur Temps Réel (Live Traffic Monitor)
- **Codes de direction** :
  - `>> [APP->DEV]` : Commandes envoyées par le logiciel vers l'appareil.
  - `<< [DEV->APP]` : Réponses envoyées par l'appareil vers le logiciel.
- **Affichage configurable** :
  - *Dual (Hex + ASCII)* : vue complète idéale pour repérer à la fois les entêtes binaires et les commandes texte.
  - *Hex Only* : idéal pour les trames binaires pures (ADC, checksums, octets de contrôle).
  - *ASCII Only* : lisibilité maximale pour les protocoles texte (SCPI / commandes AT / texte délimité).
- **Filtres & Recherche** :
  - Filtrage par direction (Tous, App->Dev uniquement, Dev->App uniquement).
  - Recherche textuelle instantanée (filtre les chaînes ou octets spécifiques).
  - Pause et Défilement automatique (Auto-scroll).

### 4.2. Enregistrement & Gestion des Sessions (Recording & Sessions)
- **Start / Stop Recording** : enregistre toutes les trames échangées dans un tampon mémoire dédié.
- **Export JSON & CSV via Explorateur de fichiers natif** :
  - Format JSON structuré complet incluant horodatage relatif, delta ms, longueur, données hexadécimales et ASCII.
  - Format CSV directement ouvrable sous Excel ou LibreOffice Calc pour tracer des chronogrammes d'échanges.
- **Chargement de session** : permet de recharger une capture précédente pour l'analyser hors-ligne ou la rejouer.

### 4.3. Bac à Sable de Replay & Injecteur (Replay & Injector Sandbox)
- **Injecteur Manuel** :
  - Envoyez des trames personnalisées en **ASCII** (ex: `?STATUS\r\n`) ou en **HEX** (ex: `02 30 31 03`).
  - Choisissez la cible : vers le matériel physique (pour tester la réaction de l'appareil à une commande) ou vers le logiciel virtuel (pour simuler une fausse réponse d'appareil).
- **Moteur de Replay Automatisé** :
  - Rejouez une séquence enregistrée vers le matériel physique.
  - **Filtrage automatique de direction** : rejoue uniquement les commandes `APP->DEV` vers l'appareil (évite d'envoyer à l'appareil ses propres réponses en écho).
  - **Timing fidèle** : reproduit les intervalles réels entre les commandes avec multiplicateur de vitesse réglable (`0.25x`, `0.5x`, `1.0x Real`, `2.0x`, `Max`).
  - **Délai minimum configurable** (par défaut 20 ms) : évite d'engorger le microcontrôleur de l'appareil récepteur.
  - Contrôle pas-à-pas (*Step*) et pause/reprise.

---

## 5. Intégration dans le Graphe de Nœuds (Pipeline)

Le module s'intègre avec les autres nœuds de l'application :

### Sorties (Outputs) :
- `received_data` (`IOTypes.TEXT`) : émet la chaîne ASCII de chaque trame sniffée.
- `packet_data` (`IOTypes.CMD_DICT`) : émet un dictionnaire complet pour chaque trame :
  ```json
  {
    "index": 42,
    "timestamp": 1.254,
    "abs_time": "14:22:05.123",
    "delta_ms": 15.2,
    "direction": "DEV->APP",
    "hex_data": "02 4F 4B 03",
    "ascii_data": ".OK.",
    "length": 4
  }
  ```

### Entrée (Input Callback) :
- Accepte une chaîne de caractères (qui sera automatiquement injectée vers l'appareil physique).
- Accepte un dictionnaire de commande :
  - `{"action": "start_bridge"}` : démarre le sniffer.
  - `{"action": "stop_bridge"}` : arrête le sniffer.
  - `{"action": "inject", "target": "DEV", "data": "COMMAND\r\n", "is_hex": false}` : injection automatisée.
