from .openai import ChatgptTranslate

load_translations()  # type: ignore


class DeepseekTranslate(ChatgptTranslate):
    name = 'DeepSeek'
    alias = 'DeepSeek (Chat)'
    endpoint = 'https://api.deepseek.com/v1/chat/completions'
    temperature = 1.0

    concurrency_limit = 5
    request_interval = 1.0
    request_timeout = 300.0

    models: list[str] = [
        'deepseek-chat',
        'deepseek-reasoner',
        'deepseek-v4-pro',
        'deepseek-v4-flash',
    ]
    model: str | None = models[0]

    def __init__(self):
        super().__init__()
        self.model = self.config.get('model', self.model)

    def get_models(self):
        return self.models
