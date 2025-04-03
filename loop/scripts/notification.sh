#!/bin/bash

# Script pour envoyer une notification via Apprise
# Usage: ./notification.sh "Votre message"

MESSAGE="$1"

apprise -b "$MESSAGE" 'pbul://o.mQa6kOD4iVyt7454JPiurfBpJLppaHtd'
