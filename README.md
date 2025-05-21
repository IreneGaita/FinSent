# FinSent
>**⚠️ Project in development ⚠️**  
> This project is currently a work in progress. Features may be incomplete or unstable, and code is subject to frequent changes.

---

Automated system for classifying sentiment of financial news, making the analysis process faster and more scalable.  
The classification focuses on three main categories: **positive**, **negative**, and **neutral**.

The ability to quickly classify financial news will enable investors to make informed decisions, saving time and energy while reducing the risk of decisions based on misinterpretations of information.

The dataset was created specifically as a benchmark for training and evaluating sentiment analysis models, with a particular focus on the **economic and financial context**.

---

## 🚀 Running the Project with Docker and Airflow

### Prerequisites

- [Git](https://git-scm.com/downloads)
- [Docker Desktop](https://www.docker.com/products/docker-desktop) installed and running

---

### 🔧 Setup Instructions

1. **Clone the repository**

   ```bash
   git clone https://github.com/IreneGaita/FinSent.git
   cd repo
   ```
2. **Make sure Docker is running**
   
   Start Docker Desktop and wait until it's fully operational.
3. **Navigate to the `airflow/`folder (if applicable)**
    ```bash
    cd airflow
    ```
4. **Build and start the containers**
   
   Run the following command:
   ```bash
    docker-compose up --build
    ```
     > Use `docker-compose up -d` to start the containers in the background.
     
### 🌐 **Access the Airflow Web Interface**
5. **Open your browser and go to:**
   ```bash
    http://localhost:8080
    ```
   > ⚠️ Check the correct port in the docker-compose.yml file under the ports: section (e.g., 8080:8080 or 8081:8080).
   
6. **Login credentials**
   Default credentials are usually:
   - **Username:** airflow
   - **Password:** airflow
     
  > ⚠️ You can confirm or override these values in the docker-compose.yml file under the environment section.
   
### 🛑 **Shutting Down the Project**
To stop the containers, run:
  ```bash
  docker-compose down
  ```
