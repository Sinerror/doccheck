# Planted-error eval

Dataset: `data/eval/` — 15 synthetic business documents (support replies, commercial proposals, internal memos) with **56 planted errors** of 9 kinds and 47 correct statements (d01–d10 basic, d11–d15 hard: arithmetic over the price list, working vs calendar days, exceptions to rules, multi-hop facts, tricky statements that are correct), checked against the synthetic corpus in `data/corpus/` (tariffs RU/KZ, offer + archived offer, SLA, support, integrations, security, migration, features). Built by `scripts/build_eval_set.py`.

## Summary by configuration

Mean ± sd over runs of the same configuration (stochastic models: each run is one sample).

| configuration | set | runs | analyst | recall strict | flagged for human | analyst r0 | caught only after debate | FP strict (correct claims) | FP → human (correct claims) | grounding | $ / document |
|---|---|---|---|---|---|---|---|---|---|---|---|
| analyst-only | all (56 errors) | 1 | `deepseek-v4.1-flash` | 96% | 100% | 96% | 0.0 | 5% | 7% | 100% | $0.015 |
| analyst-only | basic (37 errors) | 1 | `deepseek-v4.1-flash` | 100% | 100% | 100% | 0.0 | 5% | 5% | 100% | $0.015 |
| analyst-only | hard (19 errors) | 1 | `deepseek-v4.1-flash` | 90% | 100% | 90% | 0.0 | 6% | 11% | 100% | $0.014 |
| full-v1 | all (56 errors) | 1 | `deepseek-v4.1-flash` | 95% | 100% | 96% | 2.0 | 5% | 7% | 100% | $0.222 |
| full-v1 | basic (37 errors) | 1 | `deepseek-v4.1-flash` | 97% | 100% | 100% | 0.0 | 5% | 5% | 100% | $0.226 |
| full-v1 | hard (19 errors) | 1 | `deepseek-v4.1-flash` | 90% | 100% | 90% | 2.0 | 6% | 11% | 100% | $0.214 |
| full-v2 | all (56 errors) | 1 | `deepseek-v4.1-flash` | 100% | 100% | 96% | 2.0 | 2% | 2% | 100% | $0.139 |
| full-v2 | basic (37 errors) | 1 | `deepseek-v4.1-flash` | 100% | 100% | 100% | 0.0 | 2% | 2% | 100% | $0.138 |
| full-v2 | hard (19 errors) | 1 | `deepseek-v4.1-flash` | 100% | 100% | 90% | 2.0 | 0% | 0% | 100% | $0.141 |
| local-analyst | all (56 errors) | 1 | `mistralai/devstral-small-2-2512 (local)` | 93% | 100% | 93% | 0.0 | 0% | 16% | 97% | $0.000 |
| local-analyst | basic (37 errors) | 1 | `mistralai/devstral-small-2-2512 (local)` | 97% | 100% | 97% | 0.0 | 0% | 12% | 98% | $0.000 |
| local-analyst | hard (19 errors) | 1 | `mistralai/devstral-small-2-2512 (local)` | 84% | 100% | 84% | 0.0 | 0% | 22% | 96% | $0.000 |

## Latest run, full pipeline (analyst ⇄ critic): `eval-20261107-183017-full-v2`

Models: scout = `gpt-6-luna`, analyst = `deepseek-v4.1-flash`, critic = `gpt-6.1-sol` · embeddings `intfloat/multilingual-e5-large` · prompts `61d86cb3` · max_claims=10, debate_rounds=2

| metric | value |
|---|---|
| planted errors extracted as claims (Scout) | 100% |
| **recall, strict** (right verdict, or critic's position on a disputed claim) | **100%** |
| **recall, flagged for human** (error lands in the review list) | **100%** |
| recall, analyst alone (round 0) | 96% |
| caught only after the analyst ⇄ critic debate | 2 of 56 |
| false positives on correct claims, strict (CONTRADICTED / OVERSTATED) | 2% of 58 |
| correct claims sent to a human anyway | 2% |
| extra claims (not labelled) | 1 |
| grounding rate (quotes verified by code) | 100% (258/258) |
| cost (router price estimate) | $2.09 total · $0.139 per document |
| LLM calls / wall time (sum over documents) | 457 / 750.0 s |

### Recall by error kind

| kind | n | extracted | caught strict | flagged for human | analyst r0 |
|---|---|---|---|---|---|
| condition (изменённое условие) | 15 | 15 | 15 | 15 | 15 |
| object_swap (свойство другого тарифа/региона) | 9 | 9 | 9 | 9 | 9 |
| number (неверное число) | 8 | 8 | 8 | 8 | 8 |
| overstatement (преувеличение) | 6 | 6 | 6 | 6 | 6 |
| deadline (неверный срок/дата) | 5 | 5 | 5 | 5 | 5 |
| arithmetic (ошибка в расчёте) | 5 | 5 | 5 | 5 | 3 |
| outdated (из архивной редакции) | 3 | 3 | 3 | 3 | 3 |
| not_in_sources (нет в источниках) | 3 | 3 | 3 | 3 | 3 |
| wrong_clause (неверная ссылка на пункт) | 2 | 2 | 2 | 2 | 2 |

### Every planted error

| id | set | kind | planted statement | analyst r0 | final | status | caught |
|---|---|---|---|---|---|---|---|
| d01-A | basic | outdated | Согласно п. 4.1 оферты, полный возврат первой оплаты возможен, если обратиться в течение 30 календарных дней с | CONTRADICTED, CONTRADICTED | CONTRADICTED, CONTRADICTED | accepted, accepted | ✅ |
| d01-B | basic | condition | Если вы решите вернуть только неиспользованные месяцы, скидка за годовую оплату при перерасчёте сохраняется. | CONTRADICTED | CONTRADICTED | accepted | ✅ |
| d01-C | basic | number | Если вы захотите направить официальную претензию, срок ответа на неё составляет 30 рабочих дней. | CONTRADICTED | CONTRADICTED | accepted | ✅ |
| d02-A | basic | number | Тариф «Команда» рассчитан на команды до 50 пользователей, поэтому формально подходит и вам, но мы рекомендуем  | CONTRADICTED | CONTRADICTED | accepted | ✅ |
| d02-B | basic | object_swap | Кроме того, за вашей компанией на тарифе «Бизнес» будет закреплён персональный менеджер. | CONTRADICTED | CONTRADICTED | accepted | ✅ |
| d02-C | basic | object_swap | Гарантированная доступность сервиса на тарифе «Бизнес» составляет 99,9% в месяц. | CONTRADICTED | CONTRADICTED | accepted | ✅ |
| d02-D | basic | deadline | Перенос под ключ занимает не более 5 рабочих дней после получения доступа к исходной системе. | CONTRADICTED | CONTRADICTED | accepted | ✅ |
| d03-A | basic | object_swap | Для клиентов из Казахстана пробный период тарифа «Бизнес» составляет 14 дней. | CONTRADICTED | CONTRADICTED | accepted | ✅ |
| d03-B | basic | object_swap | Тариф «Команда» в Казахстане рассчитан на команды до 25 пользователей. | CONTRADICTED | CONTRADICTED | accepted | ✅ |
| d03-C | basic | condition | Если вам удобнее, счёт можно оплатить и в рублях. | CONTRADICTED | CONTRADICTED | accepted | ✅ |
| d04-A | basic | condition | Время реакции по SLA — это гарантированное время, за которое проблема будет решена. | CONTRADICTED | CONTRADICTED | accepted | ✅ |
| d04-B | basic | condition | Компенсация за недоступность выплачивается деньгами на расчётный счёт клиента. | CONTRADICTED | CONTRADICTED | accepted | ✅ |
| d04-C | basic | deadline | Плановые технические работы проводятся по субботам с 02:00 до 05:00 по московскому времени. | CONTRADICTED | CONTRADICTED | accepted | ✅ |
| d04-D | basic | overstatement | Круглосуточная поддержка по любым обращениям доступна клиентам тарифов «Бизнес» и «Корпоративный». | CONTRADICTED | CONTRADICTED | accepted | ✅ |
| d05-A | basic | overstatement | REST API доступен на всех тарифах, включая «Старт». | CONTRADICTED | CONTRADICTED | accepted | ✅ |
| d05-B | basic | condition | Если этого не хватит, лимит на тарифе «Бизнес» можно повысить за дополнительную плату. | CONTRADICTED | CONTRADICTED | accepted | ✅ |
| d05-C | basic | number | Если ваш сервер не ответит, мы повторим доставку вебхука 5 раз. | CONTRADICTED | CONTRADICTED | accepted | ✅ |
| d05-D | basic | condition | Синхронизация с 1С выполняется в реальном времени, сразу после каждого изменения. | CONTRADICTED | CONTRADICTED | accepted | ✅ |
| d06-A | basic | overstatement | Эта скидка распространяется на все тарифы, включая «Корпоративный». | CONTRADICTED | CONTRADICTED | accepted | ✅ |
| d06-B | basic | number | Скидку для образования можно совместить со скидкой за оплату на год — итоговая скидка составит 50%. | CONTRADICTED, CONTRADICTED | SUPPORTED, CONTRADICTED | revised, accepted | ✅ |
| d06-C | basic | number | Каждый дополнительный час персонального обучения стоит 3 500 ₽. | CONTRADICTED | CONTRADICTED | accepted | ✅ |
| d06-D | basic | not_in_sources | Сорока CRM включена в единый реестр российского программного обеспечения. | UNVERIFIABLE | UNVERIFIABLE | accepted | ✅ |
| d07-A | basic | object_swap | Резервные копии создаются ежедневно и на тарифе «Бизнес» хранятся 90 дней. | SUPPORTED, CONTRADICTED | SUPPORTED, CONTRADICTED | accepted, accepted | ✅ |
| d07-B | basic | object_swap | Единый вход по протоколу SAML 2.0 (SSO) доступен на тарифе «Бизнес». | CONTRADICTED | CONTRADICTED | accepted | ✅ |
| d07-C | basic | not_in_sources | Сервис сертифицирован по международному стандарту ISO/IEC 27001. | UNVERIFIABLE | UNVERIFIABLE | accepted | ✅ |
| d07-D | basic | overstatement | Отчёт о последнем тесте на проникновение мы предоставляем по запросу любому клиенту. | CONTRADICTED | CONTRADICTED | accepted | ✅ |
| d08-A | basic | outdated | Согласно п. 3.4 оферты, если оплата не поступила, доступ блокируется через 3 календарных дня после окончания о | CONTRADICTED, CONTRADICTED, CONTRADICTED | CONTRADICTED, CONTRADICTED, SUPPORTED | accepted, accepted, revised | ✅ |
| d08-B | basic | outdated | Об изменении цен тарифов мы уведомляем не менее чем за 14 календарных дней. | CONTRADICTED | CONTRADICTED | accepted | ✅ |
| d08-C | basic | wrong_clause | Согласно п. 9.2 оферты, за каждый день просрочки оплаты начисляется пеня 0,1%. | UNVERIFIABLE, UNVERIFIABLE | UNVERIFIABLE, UNVERIFIABLE | accepted, accepted | ✅ |
| d09-A | basic | object_swap | При оплате тарифа «Бизнес» сразу на два года вы получите скидку 25%. | CONTRADICTED | CONTRADICTED | accepted | ✅ |
| d09-B | basic | condition | При переносе под ключ мы также перенесём настройки автоматизаций и шаблоны документов. | CONTRADICTED | CONTRADICTED | accepted | ✅ |
| d09-C | basic | wrong_clause | Согласно разделу 6 оферты, данные казахстанских клиентов хранятся в центре обработки данных в Москве. | CONTRADICTED | CONTRADICTED | accepted | ✅ |
| d09-D | basic | number | Электронный счёт-фактура выставляется через ЭСФ в течение 5 календарных дней после оказания услуг. | CONTRADICTED | CONTRADICTED | accepted | ✅ |
| d10-A | basic | number | На тарифе «Бизнес» каждый пользователь может сделать до 500 запросов к AI-ассистенту в месяц. | CONTRADICTED | CONTRADICTED | accepted | ✅ |
| d10-B | basic | condition | В мобильном приложении можно полноценно работать офлайн: изменения синхронизируются после подключения к интерн | CONTRADICTED | CONTRADICTED | accepted | ✅ |
| d10-C | basic | object_swap | Автоматическая расшифровка звонков доступна на тарифе «Бизнес». | CONTRADICTED | CONTRADICTED | accepted | ✅ |
| d10-D | basic | not_in_sources | AI-ассистент работает на собственной модели Сорока, обученной на 10 миллионах диалогов отделов продаж. | UNVERIFIABLE | UNVERIFIABLE | accepted | ✅ |
| d11-A | hard | arithmetic | Для 120 сотрудников доплата за дополнительных пользователей составит 2 000 ₽ в месяц. | CONTRADICTED | CONTRADICTED | accepted | ✅ |
| d11-B | hard | arithmetic | Это даёт экономию 25 980 ₽ в год по сравнению с помесячной оплатой. | UNVERIFIABLE | CONTRADICTED | revised | ✅ |
| d11-C | hard | condition | Если команда вырастет до 180 человек, вы сможете остаться на тарифе «Бизнес», просто докупив пользователей. | CONTRADICTED | CONTRADICTED | accepted | ✅ |
| d11-D | hard | overstatement | Дополнительное хранилище можно докупить на любом тарифе пакетами по 100 ГБ за 590 ₽ в месяц. | OVERSTATED, SUPPORTED | OVERSTATED, SUPPORTED | accepted, accepted | ✅ |
| d12-A | hard | condition | Мы вернём стоимость неиспользованных полных месяцев по той же цене, по которой вы их оплатили, со скидкой. | CONTRADICTED | CONTRADICTED | accepted | ✅ |
| d12-B | hard | arithmetic | За два использованных месяца из оплаченной суммы будет удержано 20 784 ₽. | UNVERIFIABLE | CONTRADICTED | revised | ✅ |
| d12-C | hard | deadline | Деньги поступят в течение 10 календарных дней с даты получения заявления. | CONTRADICTED | CONTRADICTED | accepted | ✅ |
| d13-A | hard | arithmetic | Если доступность на тарифе «Команда» за месяц составила 99,2%, компенсация составит 20% месячной стоимости. | CONTRADICTED | CONTRADICTED | accepted | ✅ |
| d13-B | hard | arithmetic | При доступности 98% на тарифе «Корпоративный» клиенту положена компенсация 100% месячной стоимости. | CONTRADICTED | CONTRADICTED | accepted | ✅ |
| d13-C | hard | deadline | Запрос на компенсацию принимается в течение 30 календарных дней с даты инцидента. | CONTRADICTED | CONTRADICTED | accepted | ✅ |
| d13-D | hard | condition | В праздничные дни поддержка не работает, все обращения рассматриваются после праздников. | CONTRADICTED, CONTRADICTED | CONTRADICTED, CONTRADICTED | accepted, accepted | ✅ |
| d14-A | hard | condition | Как некоммерческая организация вы получите скидку 30% на ваш текущий тариф «Старт». | CONTRADICTED | CONTRADICTED | accepted | ✅ |
| d14-B | hard | condition | Скидку для НКО можно также совместить с акцией «Переезд». | CONTRADICTED | CONTRADICTED | accepted | ✅ |
| d14-C | hard | number | По акции «Переезд» первые 6 месяцев тарифа «Команда» стоят вдвое дешевле. | CONTRADICTED | CONTRADICTED | accepted | ✅ |
| d14-D | hard | overstatement | Интеграция с Telegram бесплатна на всех тарифах. | CONTRADICTED | CONTRADICTED | accepted | ✅ |
| d15-A | hard | condition | Продлевать пробный период можно несколько раз, пока вы не примете решение. | CONTRADICTED | CONTRADICTED | revised | ✅ |
| d15-B | hard | object_swap | Резервные копии ваших данных хранятся в центре обработки данных в Москве. | CONTRADICTED | CONTRADICTED | accepted | ✅ |
| d15-C | hard | condition | На тарифе «Команда» восстановление данных из резервной копии выполняется бесплатно раз в квартал. | CONTRADICTED | CONTRADICTED | accepted | ✅ |
| d15-D | hard | deadline | Электронный счёт-фактура выставляется через ЭСФ в течение 15 рабочих дней после оказания услуг. | CONTRADICTED | CONTRADICTED | accepted | ✅ |

### False positives (strict) in that run

- `d02_kp_logistika.md` C8 (clean): «При оплате на год перенос из текущей CRM выполняется бесплатно в объёме до 50 000 записей.» → OVERSTATED (revised)

## History

| eval | label | analyst | prompts | recall strict | flagged | r0 | FP strict | grounding | cost |
|---|---|---|---|---|---|---|---|---|---|
| `eval-20261107-180552-analyst-only` | analyst-only | `deepseek-v4.1-flash` | `642eb243` | 96% | 100% | 96% | 5% | 100% | $0.22 |
| `eval-20261107-181248-full` | full-v1 | `deepseek-v4.1-flash` | `642eb243` | 95% | 100% | 96% | 5% | 100% | $3.33 |
| `eval-20261107-183017-full-v2` | full-v2 | `deepseek-v4.1-flash` | `61d86cb3` | 100% | 100% | 96% | 2% | 100% | $2.09 |
| `eval-20261107-183945-local-analyst` | local-analyst | `mistralai/devstral-small-2-2512 (local)` | `4cc4818c` | 93% | 100% | 93% | 0% | 97% | $0.00 |

## Definitions and caveats

- *Strict* catch: the final verdict is one the label expects (CONTRADICTED/OVERSTATED for most kinds; UNVERIFIABLE also counts for statements absent from the sources and for references to clauses that do not exist), or the claim stayed disputed and the critic's suggested verdict is an expected one.
- *Flagged for human*: the claim is in the report's review list (verdict ≠ SUPPORTED, disputed, confidence < 0.6, or a failed quote). This is what matters in production: the error does not leave the company unseen.
- Claims are matched to labelled sentences by the `quote` the Scout copies from the document (fuzzy ≥ 80).
- The dataset and the corpus are synthetic and were written by the author of the system: the numbers show how the pipeline behaves on these error types, not a benchmark of real-world accuracy.
- Analyst-only rows reuse the Scout's claims of an earlier run (`--reuse-claims`), so analyst models are compared on identical claims.
