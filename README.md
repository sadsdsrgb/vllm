# vLLM Multi-Implementation Deployment

Этот проект содержит настройку и развертывание трех различных реализаций vLLM для работы с моделями из Ollama на удаленном сервере.

## 🎯 Обзор

### Три реализации vLLM:

1. **vLLM (Оригинальный)** - Production-ready решение с максимальной производительностью
2. **Nano-vLLM** - Упрощенная реализация для обучения и экспериментов
3. **oVLLM** - Простая обертка с интеграцией DSPy

### Доступные модели:
- **qwen3:30b** (30.5B параметров, Q4_K_M квантизация)
- **gpt-oss:20b** (20.9B параметров, MXFP4 квантизация)

### Распределение портов:
- **8000**: vLLM + Qwen3 30B
- **8001**: vLLM + GPT-OSS 20B  
- **8002**: Nano-vLLM + Qwen3 30B
- **8003**: Nano-vLLM + GPT-OSS 20B
- **8004**: oVLLM + Qwen3 30B
- **8005**: oVLLM + GPT-OSS 20B

## 🚀 Быстрый старт

### 1. Подготовка

```bash
# Клонирование и переход в директорию
cd vllm

# Проверка доступности Ollama
curl http://localhost:11434/api/tags

# Создание необходимых директорий
mkdir -p {vllm,nanovllm,ovllm}/{logs,models}
```

### 2. Развертывание всех сервисов

```bash
# Автоматическое развертывание всех сервисов
chmod +x scripts/deploy_all.sh
./scripts/deploy_all.sh
```

### 3. Мониторинг

```bash
# Запуск интерактивного мониторинга
chmod +x scripts/monitor.sh
./scripts/monitor.sh

# Быстрая проверка статуса
./scripts/monitor.sh --quick
```

### 4. Тестирование API

```bash
# Полное тестирование всех API
chmod +x scripts/test_apis.sh
./scripts/test_apis.sh

# Быстрая проверка подключения
./scripts/test_apis.sh --quick

# Тест производительности
./scripts/test_apis.sh --performance
```

## 📁 Структура проекта

```
vllm/
├── vllm/                          # Оригинальный vLLM
│   ├── docker-compose.yml
│   ├── Dockerfile.vllm
│   └── scripts/
│       ├── start_vllm.py
│       └── ollama_model_loader.py
├── nanovllm/                      # Nano-vLLM
│   ├── docker-compose.yml
│   ├── Dockerfile.nano
│   └── scripts/
│       ├── start_nano_vllm.py
│       ├── nano_ollama_loader.py
│       └── nano_api_server.py
├── ovllm/                         # oVLLM
│   ├── docker-compose.yml
│   ├── Dockerfile.ovllm
│   └── scripts/
│       ├── start_ovllm.py
│       ├── ovllm_ollama_loader.py
│       └── ovllm_api_server.py
├── scripts/                       # Управляющие скрипты
│   ├── deploy_all.sh             # Автоматическое развертывание
│   ├── monitor.sh                # Мониторинг сервисов
│   └── test_apis.sh              # Тестирование API
├── docker-compose.master.yml      # Главный compose файл
├── deployment_guide.md            # Подробное руководство
├── functionality_comparison.md    # Сравнение функционала
└── README.md                      # Этот файл
```

## 🔧 Использование API

### Стандартные OpenAI-совместимые endpoints

#### Chat Completions
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3:30b",
    "messages": [
      {"role": "user", "content": "Привет! Как дела?"}
    ],
    "temperature": 0.7,
    "max_tokens": 1000
  }'
```

#### Text Completions
```bash
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3:30b", 
    "prompt": "Напиши короткий рассказ о",
    "temperature": 0.7,
    "max_tokens": 500
  }'
```

### Уникальные DSPy endpoints (только oVLLM - порты 8004, 8005)

#### DSPy Predict
```bash
curl -X POST http://localhost:8004/v1/dspy/predict \
  -H "Content-Type: application/json" \
  -d '{
    "signature": "question -> answer",
    "inputs": {"question": "Что такое искусственный интеллект?"},
    "temperature": 0.7,
    "max_tokens": 500
  }'
```

#### DSPy Chain of Thought
```bash
curl -X POST http://localhost:8004/v1/dspy/chain_of_thought \
  -H "Content-Type: application/json" \
  -d '{
    "signature": "math_problem -> reasoning, answer",
    "inputs": {"math_problem": "Если у меня 5 яблок и я съел 2, сколько осталось?"},
    "temperature": 0.3,
    "max_tokens": 400
  }'
```

## 📊 Сравнение реализаций

| Характеристика | vLLM | Nano-vLLM | oVLLM |
|----------------|------|-----------|-------|
| **Производительность** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Простота** | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Функциональность** | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ |
| **Стабильность** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |

### vLLM (Порты 8000, 8001)
- ✅ Максимальная производительность (до 24x быстрее HuggingFace)
- ✅ PagedAttention для эффективного управления памятью
- ✅ Continuous Batching
- ✅ Полная совместимость с OpenAI API
- ✅ Production-ready
- ❌ Сложная настройка

### Nano-vLLM (Порты 8002, 8003)
- ✅ Простая и понятная архитектура (~1200 строк кода)
- ✅ Сравнимая производительность с vLLM
- ✅ Легко модифицировать
- ✅ Образовательная ценность
- ❌ Ограниченный функционал
- ❌ Нет streaming

### oVLLM (Порты 8004, 8005)
- ✅ Крайне простой API ("одна строка кода")
- ✅ Интеграция с DSPy
- ✅ Автоматическое батчирование
- ✅ Chain of Thought reasoning
- ❌ Зависимость от vLLM
- ❌ Ограниченная гибкость

## 🛠️ Управление сервисами

### Запуск отдельных сервисов

```bash
# Только vLLM
docker-compose -f docker-compose.master.yml up -d vllm-qwen3 vllm-gpt-oss

# Только Nano-vLLM  
docker-compose -f docker-compose.master.yml up -d nano-vllm-qwen3 nano-vllm-gpt-oss

# Только oVLLM
docker-compose -f docker-compose.master.yml up -d ovllm-qwen3 ovllm-gpt-oss
```

### Просмотр логов

```bash
# Все сервисы
docker-compose -f docker-compose.master.yml logs -f

# Конкретный сервис
docker-compose -f docker-compose.master.yml logs -f vllm-qwen3
```

### Перезапуск сервисов

```bash
# Все сервисы
docker-compose -f docker-compose.master.yml restart

# Конкретный сервис
docker-compose -f docker-compose.master.yml restart vllm-qwen3
```

### Остановка сервисов

```bash
# Все сервисы
docker-compose -f docker-compose.master.yml down

# С удалением volumes
docker-compose -f docker-compose.master.yml down -v
```

## 🔍 Мониторинг и диагностика

### Проверка здоровья всех сервисов

```bash
for port in 8000 8001 8002 8003 8004 8005; do
  echo "Port $port:"
  curl -s http://localhost:$port/health | jq .
done
```

### Мониторинг ресурсов

```bash
# GPU использование
nvidia-smi

# Использование ресурсов контейнерами
docker stats

# Интерактивный мониторинг
./scripts/monitor.sh
```

### Проверка доступных моделей в Ollama

```bash
curl http://localhost:11434/api/tags | jq '.models[].name'
```

## ⚡ Производительность и оптимизация

### Рекомендуемые настройки для разных GPU

#### GPU с 24GB+ VRAM:
```yaml
environment:
  - GPU_MEMORY_UTILIZATION=0.9
  - TENSOR_PARALLEL_SIZE=1
  - MAX_MODEL_LEN=16384
```

#### GPU с 16-24GB VRAM:
```yaml
environment:
  - GPU_MEMORY_UTILIZATION=0.8
  - TENSOR_PARALLEL_SIZE=1
  - MAX_MODEL_LEN=8192
```

#### GPU с 8-16GB VRAM:
```yaml
environment:
  - GPU_MEMORY_UTILIZATION=0.7
  - TENSOR_PARALLEL_SIZE=1
  - MAX_MODEL_LEN=4096
```

### Бенчмарки производительности

```bash
# Тест производительности всех сервисов
./scripts/test_apis.sh --performance

# Нагрузочное тестирование
./scripts/test_apis.sh --load 50 5  # 50 запросов, 5 одновременно
```

## 🐛 Устранение неполадок

### Частые проблемы

#### 1. Контейнер не запускается
```bash
# Проверить логи
docker logs <container_name>

# Проверить доступность GPU
docker run --rm --gpus all nvidia/cuda:12.1-base-ubuntu22.04 nvidia-smi
```

#### 2. Модель не найдена в Ollama
```bash
# Проверить доступные модели
curl http://localhost:11434/api/tags

# Загрузить модель если нужно
curl -X POST http://localhost:11434/api/pull -d '{"name": "qwen3:30b"}'
```

#### 3. Недостаточно GPU памяти
- Уменьшить `GPU_MEMORY_UTILIZATION`
- Уменьшить `MAX_MODEL_LEN`
- Использовать квантизацию

#### 4. API возвращает ошибки
```bash
# Проверить статус сервиса
curl http://localhost:8000/health

# Проверить логи
docker-compose -f docker-compose.master.yml logs vllm-qwen3
```

### Диагностические команды

```bash
# Полная диагностика
./scripts/monitor.sh --quick

# Тестирование API
./scripts/test_apis.sh --quick

# Проверка Docker
docker-compose -f docker-compose.master.yml ps
```

## 📚 Дополнительная документация

- [deployment_guide.md](deployment_guide.md) - Подробное руководство по развертыванию
- [functionality_comparison.md](functionality_comparison.md) - Детальное сравнение функционала

## 🤝 Поддержка

### Ссылки на оригинальные проекты:
- [vLLM](https://github.com/vllm-project/vllm) - Apache-2.0 License
- [Nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm) - MIT License  
- [oVLLM](https://github.com/MaximeRivest/ovllm) - MIT License

### Полезные команды для отладки:

```bash
# Проверка всех сервисов
./scripts/monitor.sh --quick

# Полное тестирование
./scripts/test_apis.sh

# Просмотр логов всех сервисов
docker-compose -f docker-compose.master.yml logs --tail=100

# Перезапуск проблемного сервиса
docker-compose -f docker-compose.master.yml restart <service_name>
```

## 🎉 Заключение

Все три реализации vLLM успешно настроены и готовы к работе с вашими моделями из Ollama:

- **Порты 8000-8001**: vLLM для production нагрузок
- **Порты 8002-8003**: Nano-vLLM для экспериментов  
- **Порты 8004-8005**: oVLLM для простых задач и DSPy

Каждая реализация имеет свои преимущества и подходит для разных сценариев использования. Выберите наиболее подходящую для ваших задач!
