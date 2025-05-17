Scripts disponíveis:

- fr_inference.py: Usado para testar a implementação como um todo, tem as partes de reconhecer um rosto e salvar seus embeddings caso tenha uma pessoa na imagem e reconhecer essa pessoa no meio de uma multidão, para trocar entre os módulos de execução é necessário chamar o service: ros2 service call /toggle_mode std_srvs/srv/Empty;
- fr_node.py: Ativa o nó para o módulo de reconhecimento de faces (atualmente usado na task de recepcionista);
- fr_run_task: Usado para executar a task de reconhecimento de pessoas;
