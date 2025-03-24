# AutoQuest

AutoQuest é um assistente amigável especializado em manuais de veículos, desenvolvido utilizando Flask, Firestore e Vertex AI. Este projeto permite que os usuários façam perguntas sobre manuais de veículos e obtenham respostas precisas com base em um banco de dados vetorial.

## Pré-requisitos

- Python 3.8 ou superior
- Conta no Google Cloud com acesso ao Firestore e Vertex AI
- Credenciais de autenticação do Google Cloud configuradas

## Instalação

1. Clone o repositório:

    ```bash
    git clone https://github.com/seu-usuario/autoquest.git
    cd autoquest
    ```

2. Crie um ambiente virtual e ative-o:

    ```bash
    python -m venv venv
    source venv/bin/activate  # No Windows, use `venv\Scripts\activate`
    ```

3. Instale as dependências:

    ```bash
    pip install -r requirements.txt
    ```

4. Configure as credenciais do Google Cloud:

    ```bash
    export GOOGLE_APPLICATION_CREDENTIALS="path/to/your/credentials.json"
    ```

## Configuração

1. Atualize o arquivo `main.py` com o ID do seu projeto do Google Cloud:

    ```python
    PROJECT_ID = "seu-id-do-projeto"
    ```

2. Certifique-se de que a coleção Firestore `cod-civil2` está configurada corretamente no seu projeto do Google Cloud.

## Executando o Projeto

1. Inicie o servidor Flask:

    ```bash
    python main.py
    ```

2. Acesse o aplicativo em seu navegador:

    ```
    http://localhost:8080
    ```

## Uso

- Acesse a página inicial e faça uma pergunta sobre manuais de veículos.
- O AutoQuest buscará o contexto relevante no banco de dados vetorial e utilizará o modelo generativo do Vertex AI para fornecer uma resposta precisa.

## Estrutura do Código

- `main.py`: Arquivo principal que configura o Flask, Firestore, Vertex AI e define as rotas.
- `routes/main_routes.py`: Define as rotas do aplicativo.
- `templates/`: Contém os arquivos HTML para renderização das páginas.
- `static/`: Contém arquivos estáticos como favicon.



## Licença

Este projeto está licenciado sob a Licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.
