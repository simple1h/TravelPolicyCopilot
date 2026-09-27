# Динамическая архитектура Travel Policy Copilot

**Проект:** Travel Policy Copilot  
**Целевая инициатива:** Travel AI Platform  
**Тип документа:** динамическая архитектура — граф состояний LangGraph и диаграммы последовательности  

---

## 1. Назначение документа

Документ описывает поведение Travel Policy Copilot во времени и дополняет статические представления C1–C3. Он фиксирует:

- структуру сценария LangGraph;
- прерывание и возобновление работы, допустимые действия пользователя;
- основной успешный сценарий;
- уточнение неполного запроса;
- восстановление при отсутствии вариантов и ошибках поставщика;
- обработку ситуации, когда все варианты нарушают политику;
- выбор, изменение выбора, подтверждение и отказ;
- защищённый идемпотентный путь записи;
- инварианты безопасности при выполнении;

Документ использует имена компонентов из `04-agent-component-architecture.md`. 

---


## 2. Участники динамических сценариев

| Участник                            | Уровень               | Роль                                                                                               |
| ----------------------------------- | --------------------- | -------------------------------------------------------------------------------------------------- |
| Пользователь                        | Действующее лицо      | Вводит запрос, уточняет параметры, выбирает вариант и подтверждает или отклоняет запись            |
| Web UI                              | Контейнер C2          | Отправляет запросы сеанса и действия и отображает текущее состояние                                |
| Session API Controller              | Компонент C3          | Проверка запросов и граница HTTP                                                                   |
| Agent Application Service           | Компонент C3          | Управление сеансом, проверка доступа и действий, запуск и возобновление графа, формирование ответа |
| Session & Checkpoint Manager        | Компонент C3          | Изменяемое состояние сеанса и контрольной точки в памяти процесса                                  |
| LangGraph Orchestrator              | Компонент C3          | Узлы, условные переходы, прерывания и конечные состояния                                           |
| Workflow Node Handlers              | Компонент C3          | Шаги сценария приложения                                                                           |
| LLM Gateway                         | Компонент C3          | Извлечение параметров, уточнение и объяснение                                                      |
| Tool Execution Boundary             | Компонент C3          | Вызов разрешённых типизированных инструментов                                                      |
| Profile / Travel / RAG adapters     | Компонент C3s         | Интеграции для чтения                                                                              |
| Policy & Budget Engine              | Компонент C3          | Детерминированные результаты проверки                                                              |
| Recommendation Scoring              | Компонент C3          | Детерминированное ранжирование                                                                     |
| Approval Draft Writer               | Компонент C3          | Идемпотентная запись                                                                               |
| Observability Adapter               | Компонент C3          | Передача записей выполнения, событий и показателей без влияния на основной сценарий                |
| PostgreSQL / Qdrant / external APIs | Зависимости уровня C2 | Сохраняемые данные и внешние возможности                                                           |

---

## 3. Структура выполнения LangGraph

Диаграмма ниже показывает узлы и переходы графа. 

```plantuml
@startuml langgraph-execution-topology
title Travel Policy Copilot — Топология исполняемого графа LangGraph
hide empty description

state parseRequest
state validateRequest
state askClarification
state getEmployeeProfile
state searchTravelOptions
state proposeConstraintChange
state retrievePolicy
state checkPolicy
state checkBudget
state rankOptions
state presentRecommendations
state requestConfirmation
state createApprovalRequest
state showResult

[*] --> parseRequest
parseRequest --> validateRequest

validateRequest --> askClarification : clarification_required
askClarification --> [*] : awaiting user action\nEND current invocation

validateRequest --> getEmployeeProfile : request_complete
getEmployeeProfile --> searchTravelOptions

searchTravelOptions --> proposeConstraintChange : options_not_found\nor supplier_error
searchTravelOptions --> retrievePolicy : options_found

retrievePolicy --> checkPolicy
checkPolicy --> checkBudget
checkBudget --> rankOptions
rankOptions --> presentRecommendations

presentRecommendations --> [*] : awaiting_selection\nEND current invocation
presentRecommendations --> proposeConstraintChange : no_recommendation\nor all_policy_fail
presentRecommendations --> requestConfirmation : selection_made

proposeConstraintChange --> validateRequest : update_constraints
proposeConstraintChange --> searchTravelOptions : retry_search
proposeConstraintChange --> [*] : cancel

requestConfirmation --> [*] : awaiting_confirmation\nEND current invocation
requestConfirmation --> presentRecommendations : change_selection
requestConfirmation --> [*] : rejected
requestConfirmation --> createApprovalRequest : confirmed

createApprovalRequest --> showResult
showResult --> [*]

@enduml
```


## 4. Модель взаимодействующих состояний и действий

| Состояние взаимодействия / узел | Допустимые действия | Следующая точка | Запись разрешена |
|---|---|---|---|
| `ask_clarification` | `submit_clarification` | `validateRequest` после обновления состояния запроса | Нет |
| `propose_constraint_change` | `update_constraints` | `validateRequest` | Нет |
| `propose_constraint_change` | `retry_search` | `searchTravelOptions` | Нет |
| `propose_constraint_change` | `cancel` | Отменено | Нет |
| `present_recommendations` | `select_option` | `requestConfirmation` | Нет |
| `request_confirmation` | `confirm` | `createApprovalRequest` после повторных проверок | Да, только в этой ветви |
| `request_confirmation` | `reject` | Отклонено | Нет |
| `request_confirmation` | `change_selection` | `presentRecommendations` | Нет |

---

## 5. Сценарий 1 — создание сеанса и основная последовательность подготовки рекомендации

Диаграмма разделяет жизненный цикл сеанса HTTP и вычислительный сценарий. Запись в этом сценарии ещё не выполняется.

```plantuml
@startuml seq-main-recommendation-path
title Сценарий 1 — Основной путь подготовки рекомендаций
autonumber

actor User
participant "Web UI" as UI
participant "Session API\nController" as API
participant "Agent Application\nService" as APP
participant "Session & Checkpoint\nManager" as STATE
participant "LangGraph\nOrchestrator" as GRAPH
participant "Workflow Nodes" as NODES

User -> UI : Выбирает пользователя\nи вводит запрос коммандировки
UI -> API : POST /v1/agent/sessions\n(employee_id, request_text)
API -> API : Validate transport schema
API -> APP : create_session(command)
APP -> APP : Input/access/runtime guards
APP -> STATE : Create session_id/thread_id\nand initial state
STATE --> APP : Session context
APP -> GRAPH : invoke(initial state, thread_id)

GRAPH -> NODES : parseRequest
NODES --> GRAPH : Structured/normalized request
GRAPH -> NODES : validateRequest

alt request complete
    GRAPH -> NODES : getEmployeeProfile
    GRAPH -> NODES : searchTravelOptions
    GRAPH -> NODES : retrievePolicy
    GRAPH -> NODES : checkPolicy
    GRAPH -> NODES : checkBudget
    GRAPH -> NODES : rankOptions
    GRAPH -> NODES : presentRecommendations
    NODES --> GRAPH : recommendations + citations\nawaiting_selection
else request incomplete
    GRAPH -> NODES : askClarification
    NODES --> GRAPH : question + awaiting_clarification
end

GRAPH -> STATE : Save checkpoint/current node/actions
GRAPH --> APP : Graph result
APP -> APP : Build safe response projection
APP --> API : Session response + trace_id
API --> UI : 200 session state
UI --> User : Рекомендации или уточняющий вопрос

@enduml
```

### Инварианты

- Controller не обращается к внешним adapters или storage напрямую.
- `session_id`, `thread_id` и данные сотрудника связываются до вызова графа.
- Неполный запрос не запускает поиск вариантов и путь записи.
- Рекомендации формируется только после получения профиля, вариантов, оценки политик и формальных проверок.

---

## 6. Сценарий 2 — внутреннее формирование рекомендаций

Диаграмма раскрывает взаимодействия компонентов внутри node pipeline, скрытые в Sequence 1.

```plantuml
@startuml seq-recommendation-pipeline
title Сценарий 2 — Поток вычисления рекомендаций
autonumber

participant "Workflow Nodes" as NODES
participant "LLM Gateway" as LLM
participant "Tool Execution\nBoundary" as TOOLS
participant "Read-side Adapters\nProfile / Travel / RAG" as READ
participant "Policy & Budget\nEngine" as POLICY
participant "Recommendation\nScoring" as SCORE
participant "Observability\nAdapter" as OBS

NODES -> LLM : structured extraction(request_text)
LLM --> NODES : validated TripRequest\nor normalized provider error
NODES -> OBS : extraction event + usage/cost

NODES -> TOOLS : profile_lookup(employee_id)
TOOLS -> READ : Employee Profile Adapter
READ --> TOOLS : EmployeeProfile
TOOLS --> NODES : typed profile result

NODES -> TOOLS : travel_search(TripRequest)
TOOLS -> READ : Travel Search Adapter
READ --> TOOLS : offers / empty / supplier_error
TOOLS --> NODES : normalized outcome

alt offers found
    NODES -> TOOLS : policy_retrieval(query, metadata)
    TOOLS -> READ : RAG Service
    READ --> TOOLS : sanitized chunks + citations
    TOOLS --> NODES : policy evidence

    NODES -> TOOLS : policy_budget_check(profile, trip, offers, evidence)
    TOOLS -> POLICY : deterministic evaluation
    POLICY --> TOOLS : verdicts + budget results
    TOOLS --> NODES : typed decision results

    NODES -> SCORE : rank(validated offers, verdicts, preferences)
    SCORE --> NODES : ordered list + recommended option
    NODES -> LLM : explanation(summary inputs, citations)
    LLM --> NODES : explanation or safe fallback summary
    NODES -> OBS : retrieval/decision/recommendation events
else empty or supplier error
    NODES -> OBS : recovery outcome
    NODES --> NODES : Route to proposeConstraintChange
end

@enduml
```

### Инварианты

- LLM извлечение проходит типизированную проверку.
- Tool Boundary разрешает только зарегистрированные инструменты.
- Полученный контент считается данными, а не управляющими инструкциями.
- Решения по политикам/бюджету вычисляется детерминированно.
- LLM объяснение не может изменить решение.

---

## 7. Сценарий 3 — уточнение, остановка и возобновление

```plantuml
@startuml seq-clarification-resume
title Сценарий 3 — Уточненение, остановка и возобновление
autonumber

actor User
participant "Web UI" as UI
participant "Session API\nController" as API
participant "Agent Application\nService" as APP
participant "Session & Checkpoint\nManager" as STATE
participant "LangGraph\nOrchestrator" as GRAPH

GRAPH -> STATE : Save state at askClarification
GRAPH --> APP : awaiting_clarification + question
APP --> API : Session response + allowed action
API --> UI : question, action=submit_clarification
UI --> User : Показывает уточняющий вопрос

User -> UI : Вводит недостающие данные
UI -> API : POST /sessions/{id}/actions\nsubmit_clarification(payload)
API -> APP : apply_action(session_id, employee_id, action)
APP -> STATE : Load session/checkpoint
STATE --> APP : Current state=ask_clarification
APP -> APP : Verify ownership\nand action allowed; validate payload
APP -> STATE : Merge clarification into request state
APP -> GRAPH : resume(thread_id, updated state)
GRAPH -> GRAPH : validateRequest

alt request now complete
    GRAPH -> GRAPH : Continue getEmployeeProfile → ...
else still incomplete
    GRAPH -> STATE : Save new clarification checkpoint
    GRAPH --> APP : New question
end

APP --> API : Updated session state
API --> UI : Response
UI --> User : Рекомендации или следующий вопрос

@enduml
```

### Инварианты

- Ответ применяется к существующей session, а не создаёт новую.
- До заполнения обязательных параметров не вызываются profile/travel/RAG/write инструменты.

---

## 8. Сценарий 4 — восстановление при отсутствии вариантов и ошибке поставщика

```plantuml
@startuml seq-search-recovery
title Сценарий 4 — Восстановление при отсутствии вариантов и ошибке поставщика
autonumber

actor User
participant "Web UI" as UI
participant "Session API\nController" as API
participant "Agent Application\nService" as APP
participant "LangGraph\nOrchestrator" as GRAPH
participant "Travel Search\nAdapter" as TRAVEL
participant "Session & Checkpoint\nManager" as STATE

GRAPH -> TRAVEL : search(TripRequest)

alt empty offers
    TRAVEL --> GRAPH : options_not_found
else 4xx / 5xx / timeout
    TRAVEL --> GRAPH : normalized supplier_error
end

GRAPH -> GRAPH : proposeConstraintChange
GRAPH -> STATE : Save recovery state + allowed actions
GRAPH --> APP : update_constraints / retry_search / cancel
APP --> API : Recovery response
API --> UI : Recovery response
UI --> User : Предлагает изменить ограничения,\nповторить поиск или отменить

alt update_constraints
    User -> UI : Новые ограничения
    UI -> API : update_constraints action
    API -> APP : Validated action command
    APP -> STATE : Validate state/action and merge constraints
    APP -> GRAPH : resume → validateRequest
else retry_search
    User -> UI : Повторить
    UI -> API : retry_search action
    API -> APP : Validated action command
    APP -> STATE : Validate current state/action
    APP -> GRAPH : resume → searchTravelOptions
else cancel
    User -> UI : Отменить
    UI -> API : cancel action
    API -> APP : Validated action command
    APP -> STATE : Mark terminal cancelled
    APP --> API : Cancelled, no write
    API --> UI : Cancelled, no write
end

@enduml
```

### Правила восстановления

- Ошибка сервиса поиска и бронирования не преобразуется в пустой успешный набор рекомендаций.
- `retry_search` не изменяет ограничения автоматически.
- `update_constraints` возвращает поток к проверке, а не напрямую к сохранению заявки.

---

## 9. Сценарий 5 — все варианты противоречат правилам

```plantuml
@startuml seq-all-policy-fail
title Сценарий 5 - Все варианты противоречат правилам
autonumber

participant "Workflow Nodes" as NODES
participant "Policy & Budget\nEngine" as POLICY
participant "Recommendation\nScoring" as SCORE
participant "LangGraph\nOrchestrator" as GRAPH
participant "Session & Checkpoint\nManager" as STATE

NODES -> POLICY : evaluate(all offers, profile, policy evidence)
POLICY --> NODES : Every option verdict=fail\nor no eligible recommendation
NODES -> SCORE : rank with fail exclusions
SCORE --> NODES : no_recommendation
NODES --> GRAPH : no_recommendation / all_policy_fail
GRAPH -> GRAPH : proposeConstraintChange
GRAPH -> STATE : Save recovery state
GRAPH --> STATE : allowed actions = update/retry/cancel

@enduml
```

### Инварианты

- Вариант с формальным `fail` не становится рекомендованным из-за цены или удобства.
- Наличие вариантов поставщика не означает наличие допустимой рекомендации.

---

## 10. Сценарий 6 — выбор, подтверждение, отказ и идемпотентная запись

```plantuml
@startuml seq-guarded-write
title Сценарий 6 — Выбор, подтверждение, отказ и идемпотентная запись
autonumber

actor User
participant "Web UI" as UI
participant "Session API\nController" as API
participant "Agent Application\nService" as APP
participant "Session & Checkpoint\nManager" as STATE
participant "LangGraph\nOrchestrator" as GRAPH
participant "Approval Draft\nWriter" as WRITE
database "Application\nPostgreSQL" as DB

User -> UI : Выбирает option
UI -> API : select_option(session_id, employee_id, option_id)
API -> APP : Validated action command
APP -> STATE : Load session and recommendation set
STATE --> APP : awaiting_selection
APP -> APP : Verify ownership + option membership
APP -> STATE : Save selected option
APP -> GRAPH : resume → requestConfirmation
GRAPH -> STATE : Save awaiting_confirmation
APP --> API : Confirmation summary
API --> UI : Confirmation summary

alt change_selection
    User -> UI : Изменить выбор
    UI -> API : change_selection
    API -> APP : Validated action command
    APP -> STATE : Validate payload/selection
    APP -> GRAPH : resume → presentRecommendations
    APP --> API : Recommendation list
    API --> UI : Recommendation list
else reject
    User -> UI : Отказаться
    UI -> API : reject
    API -> APP : Validated action command
    APP -> STATE : Validate payload
    APP -> GRAPH : resume → terminal rejected
    APP --> API : Rejected; no approval_request_id
    API --> UI : Rejected; no approval_request_id
else confirm
    User -> UI : Подтвердить создание черновика
    UI -> API : confirm
    API -> APP : Validated action command
    APP -> STATE : Load authoritative state
    STATE --> APP : selected option + confirmation state
    APP -> APP : Re-check ownership, option, confirmation\nand write prerequisites; pre-resume\nstate/action check
    APP -> GRAPH : resume confirmed branch
    GRAPH -> WRITE : createApprovalRequest(state)
    WRITE -> WRITE : Build deterministic idempotency key\n+validate option and confirmation again
    WRITE -> DB : Find/insert by idempotency key
    alt first confirmation
        DB --> WRITE : New draft_created + request_id
    else duplicate/retry
        DB --> WRITE : Existing draft + same request_id
    end
    WRITE --> GRAPH : approval_request_id
    GRAPH -> GRAPH : showResult
    GRAPH -> STATE : Save terminal result
    APP --> API : request_id + status + trace_id
    API --> UI : request_id + status + trace_id
    UI --> User : Черновик создан
end

@enduml
```

### Инварианты

1. `select_option` не создаёт черновик.
2. `change_selection` возвращает пользователя к recommendation state.
3. `reject` заканчивает сценарий без записи.
4. `option_id` повторно проверяется непосредственно перед записью.
5. `createApprovalRequest` достижим только из ветки подтверждения.
6. Ключ идемпотентности определен для одной логической операции.
7. Повторный `confirm` не создаёт вторую строку.
8. Ограничение уникальности в БД является второй линией защиты.
9. Ошибка Langfuse не откатывает успешный запись данных приложения и не инициирует повторную запись.

---
