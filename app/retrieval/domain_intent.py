from dataclasses import dataclass
from app.retrieval.learned_intent_classifier import LearnedIntentClassifier


@dataclass(frozen=True)
class DomainIntent:
    name: str
    allowed_sources: tuple[str, ...]
    should_refuse: bool = False


SOURCE_PASSPORT = "data/raw/official_sources/citizen_passport_service_china.md"
SOURCE_LOST_PASSPORT = "data/raw/official_sources/lost_passport_return_certificate.md"
SOURCE_REGISTRATION = "data/raw/official_sources/consular_registration_china.md"
SOURCE_BIRTH = "data/raw/official_sources/birth_registration_abroad.md"
SOURCE_NOTARIAL = "data/raw/official_sources/notarial_actions_china.md"
SOURCE_LEGALIZATION = "data/raw/official_sources/document_legalization.md"
SOURCE_VISA = "data/raw/official_sources/foreign_citizen_visa_service_china.md"
SOURCE_FEES = "data/raw/official_sources/fees_and_processing_times.md"
SOURCE_SAFETY = "data/raw/official_sources/escalation_and_safety_policy.md"
SOURCE_OVERVIEW = "data/raw/official_sources/china_embassy_overview.md"


LEARNED_INTENT_CONFIDENCE_THRESHOLD = 0.15

LEARNED_INTENT_NAME_MAP = {
    "lost_passport_return_certificate": "lost_passport_return_certificate",
    "passport_service": "passport",
    "birth_registration": "birth_registration",
    "consular_registration": "consular_registration",
    "visa_service": "visa",
    "document_legalization": "document_legalization",
    "notarial_actions": "notarial_actions",
    "fees_processing": "fees_processing",
    "embassy_overview": "embassy_overview",
    "safety_escalation": "safety_escalation",
    "out_of_scope": "out_of_scope",
    "unknown": "unknown",
}

TECHNICAL_NON_CONSULAR_TERMS = (
    "endpoint",
    "api endpoint",
    "function",
    "class",
    "unit test",
    "pytest",
    "database migration",
    "software",
    "code",
)

CONSULAR_ANCHOR_TERMS = (
    "kazakhstan",
    "kazakh",
    "consular",
    "consulate",
    "embassy",
    "passport",
    "visa",
    "birth",
    "certificate",
    "return",
    "notarial",
    "legalization",
    "china",
    "китай",
    "казахстан",
    "консул",
    "посольств",
    "паспорт",
    "виза",
    "рождение",
    "доверенность",
    "легализац",
)


LEARNED_INTENT_SOURCE_MAP = {
    "lost_passport_return_certificate": (SOURCE_LOST_PASSPORT,),
    "passport_service": (SOURCE_PASSPORT,),
    "birth_registration": (SOURCE_BIRTH,),
    "consular_registration": (SOURCE_REGISTRATION,),
    "visa_service": (SOURCE_VISA,),
    "document_legalization": (SOURCE_LEGALIZATION,),
    "notarial_actions": (SOURCE_NOTARIAL,),
    "fees_processing": (SOURCE_FEES,),
    "embassy_overview": (SOURCE_OVERVIEW,),
    "safety_escalation": (SOURCE_SAFETY,),
    "out_of_scope": tuple(),
    "unknown": tuple(),
}

_LEARNED_CLASSIFIER_CACHE = None


def _get_learned_classifier():
    """Lazy-load supervised intent classifier.

    If the model artifact is missing, the router falls back to the existing
    deterministic safety/rule layer instead of crashing the API.
    """

    global _LEARNED_CLASSIFIER_CACHE

    if _LEARNED_CLASSIFIER_CACHE is not None:
        return _LEARNED_CLASSIFIER_CACHE

    try:
        _LEARNED_CLASSIFIER_CACHE = LearnedIntentClassifier()
    except (FileNotFoundError, RuntimeError, ModuleNotFoundError):
        _LEARNED_CLASSIFIER_CACHE = False

    return _LEARNED_CLASSIFIER_CACHE


def _classify_with_learned_router(question: str) -> DomainIntent | None:
    classifier = _get_learned_classifier()

    if not classifier:
        return None

    q = question.lower()

    if any(term in q for term in TECHNICAL_NON_CONSULAR_TERMS) and not any(
        term in q for term in CONSULAR_ANCHOR_TERMS
    ):
        return None

    prediction = classifier.predict(question)

    if prediction.confidence < LEARNED_INTENT_CONFIDENCE_THRESHOLD:
        return None

    if prediction.intent == "out_of_scope":
        return DomainIntent(
            name="out_of_scope",
            allowed_sources=tuple(),
            should_refuse=True,
        )

    if prediction.intent == "unknown":
        return DomainIntent(
            name="unknown",
            allowed_sources=tuple(),
            should_refuse=False,
        )

    if prediction.intent == "safety_escalation":
        return DomainIntent(
            name=LEARNED_INTENT_NAME_MAP.get(prediction.intent, prediction.intent),
            allowed_sources=LEARNED_INTENT_SOURCE_MAP[prediction.intent],
            should_refuse=False,
        )

    allowed_sources = LEARNED_INTENT_SOURCE_MAP.get(prediction.intent)

    if not allowed_sources:
        return None

    return DomainIntent(
        name=LEARNED_INTENT_NAME_MAP.get(prediction.intent, prediction.intent),
        allowed_sources=allowed_sources,
        should_refuse=False,
    )



def _contains_any(text: str, patterns: tuple[str, ...]) -> bool:
    return any(pattern in text for pattern in patterns)


def classify_domain_intent(question: str) -> DomainIntent:
    q = question.lower().strip()

    hard_safety_phrases = (
        "forged",
        "fake document",
        "false document",
        "bypass",
        "skip required documents",
        "guarantee approval",
        "promise approval",
        "lie to the consulate",
        "поддель",
        "ложн",
        "недостоверн",
        "обойти",
        "гарантировать одобрение",
        "знаком",
    )

    if any(phrase in q for phrase in hard_safety_phrases):
        return DomainIntent(
            name="safety_escalation",
            allowed_sources=LEARNED_INTENT_SOURCE_MAP.get("safety_escalation", tuple()),
            should_refuse=False,
        )

    learned_intent = _classify_with_learned_router(question)
    if learned_intent is not None:
        return learned_intent

    # Final learned-router policy:
    # If the supervised classifier exists but is not confident enough, abstain.
    # Do not fall through to broad legacy keyword rules, because clean holdout
    # evaluation showed that legacy fallback created many wrong-source errors.
    if _get_learned_classifier():
        return DomainIntent(
            name="unknown",
            allowed_sources=tuple(),
            should_refuse=False,
        )


    # High-precision lost-passport override.
    # This catches phrases like:
    # "I lost my Kazakhstan passport in China"
    # where simple phrase matching for "lost passport" fails.
    has_passport_or_identity_doc = any(
        token in q
        for token in (
            "passport",
            "identity document",
            "travel document",
            "паспорт",
            "документ",
            "удостоверение",
        )
    )

    has_lost_stolen_or_expired = any(
        token in q
        for token in (
            "lost",
            "lose",
            "stolen",
            "theft",
            "expired",
            "expire",
            "loss",
            "украли",
            "краж",
            "потер",
            "утер",
            "истек",
            "истёк",
            "просроч",
        )
    )

    asks_certificate_of_return = any(
        phrase in q
        for phrase in (
            "certificate of return",
            "return certificate",
            "свидетельство на возвращение",
            "свидетельство для возвращения",
            "вернуться в казахстан",
            "возвращение в казахстан",
        )
    )

    if (has_passport_or_identity_doc and has_lost_stolen_or_expired) or asks_certificate_of_return:
        return DomainIntent(
            name="lost_passport_return_certificate",
            allowed_sources=(SOURCE_LOST_PASSPORT,),
            should_refuse=False,
        )

    # High-precision birth-registration override.
    # This prevents birth-registration questions from falling through to generic
    # vector retrieval, which can pollute citations with passport, lost-passport,
    # registration, or embassy-overview sources.
    birth_phrases = (
        "birth registration",
        "register birth",
        "register the birth",
        "registration of birth",
        "birth of a child",
        "child birth abroad",
        "child born abroad",
        "born abroad",
        "medical birth certificate",
        "birth certificate abroad",
        "register a child",
        "register my child",
        "зарегистрировать рождение",
        "регистрация рождения",
        "рождение ребенка",
        "рождение ребёнка",
        "ребенок родился",
        "ребёнок родился",
        "родился за рубежом",
        "родилась за рубежом",
        "медицинское свидетельство о рождении",
        "свидетельство о рождении за рубежом",
    )

    if any(phrase in q for phrase in birth_phrases):
        return DomainIntent(
            name="birth_registration",
            allowed_sources=(SOURCE_BIRTH,),
            should_refuse=False,
        )

    """
    Lightweight multilingual service-intent router.

    It improves source precision for the product pipeline.
    It supports English and Russian service questions for the demo scope.
    """

    q = question.lower().strip()

    if _contains_any(
        q,
        (
            "hotel booking",
            "hotel refund",
            "booking refund",
            "university admission essay",
            "write my essay",
            "income tax",
            "tax refund",
            "personal income tax",
            "отель",
            "бронь отеля",
            "возврат за отель",
            "эссе",
            "сочинение",
            "подоходный налог",
            "налоговый возврат",
        ),
    ):
        return DomainIntent(
            name="out_of_scope",
            allowed_sources=(),
            should_refuse=True,
        )

    if _contains_any(
        q,
        (
            "replace a consular officer",
            "replace consular officer",
            "escalate",
            "guarantee",
            "guaranteed",
            "legal decision",
            "legal conclusion",
            "bypass",
            "false document",
            "forged",
            "fraud",
            "detention",
            "death",
            "safety threat",
            "sensitive case",
            "заменить сотрудника",
            "заменить консула",
            "эскалация",
            "передать сотруднику",
            "гарантировать",
            "юридическое решение",
            "юридическое заключение",
            "обойти",
            "поддель",
            "фальшив",
            "мошеннич",
            "задержан",
            "задержание",
            "смерть",
            "угроза",
            "безопасность",
        ),
    ):
        return DomainIntent(
            name="safety_escalation",
            allowed_sources=(SOURCE_SAFETY,),
        )

    if _contains_any(
        q,
        (
            "lost passport",
            "passport was lost",
            "passport is lost",
            "stolen passport",
            "passport was stolen",
            "certificate of return",
            "return certificate",
            "general travel document",
            "third countries",
            "one-time entry",
            "потерял паспорт",
            "потеряла паспорт",
            "потерян паспорт",
            "утерян паспорт",
            "украли паспорт",
            "кража паспорта",
            "свидетельство на возвращение",
            "вернуться в казахстан",
            "документ потерян",
            "документ утерян",
            "разовый въезд",
            "третьи страны",
        ),
    ):
        return DomainIntent(
            name="lost_passport_return_certificate",
            allowed_sources=(SOURCE_LOST_PASSPORT,),
        )

    if _contains_any(
        q,
        (
            "visa",
            "foreign citizen",
            "foreign applicant",
            "visa application",
            "visa fee",
            "paid visa fee",
            "working days",
            "invitation",
            "valid foreign passport",
            "work permit",
            "business visa",
            "medical insurance",
            "criminal record",
            "виза",
            "визу",
            "визовый",
            "визового",
            "визовый сбор",
            "иностранный гражданин",
            "иностранцу",
            "иностранному гражданину",
            "анкета",
            "приглашение",
            "рабочая виза",
            "бизнес-виза",
            "медицинская страховка",
            "несудимость",
        ),
    ):
        return DomainIntent(
            name="visa",
            allowed_sources=(SOURCE_VISA,),
        )

    if _contains_any(
        q,
        (
            "temporary consular registration",
            "permanent consular registration",
            "consular registration",
            "registration card",
            "consular account",
            "консульский учет",
            "консульский учёт",
            "временный учет",
            "временный учёт",
            "постоянный учет",
            "постоянный учёт",
            "учетная карточка",
            "учётная карточка",
        ),
    ):
        return DomainIntent(
            name="consular_registration",
            allowed_sources=(SOURCE_REGISTRATION,),
        )

    if _contains_any(
        q,
        (
            "birth registration",
            "register birth",
            "medical birth certificate",
            "child born",
            "birth abroad",
            "регистрация рождения",
            "зарегистрировать рождение",
            "свидетельство о рождении",
            "медицинское свидетельство",
            "ребенок родился",
            "ребёнок родился",
            "рождение за рубежом",
        ),
    ):
        return DomainIntent(
            name="birth_registration",
            allowed_sources=(SOURCE_BIRTH,),
        )

    if _contains_any(
        q,
        (
            "notarial",
            "notary",
            "power of attorney",
            "certification of copies",
            "certification of signatures",
            "certification of translations",
            "inheritance",
            "нотариаль",
            "нотариус",
            "доверенность",
            "заверить копию",
            "заверение копии",
            "засвидетельствовать подпись",
            "заверить перевод",
            "наследство",
        ),
    ):
        return DomainIntent(
            name="notarial_actions",
            allowed_sources=(SOURCE_NOTARIAL,),
        )

    if _contains_any(
        q,
        (
            "legalization",
            "legalisation",
            "apostille",
            "authenticity of signatures",
            "authenticity of seals",
            "signatures and seals",
            "легализация",
            "легализовать",
            "апостиль",
            "подписи и печати",
            "подлинность подписи",
            "подлинность печати",
        ),
    ):
        return DomainIntent(
            name="document_legalization",
            allowed_sources=(SOURCE_LEGALIZATION,),
        )

    if _contains_any(
        q,
        (
            "passport service",
            "passport processing",
            "passport renewal",
            "renew passport",
            "kazakhstan passport",
            "passport photos",
            "3.5 by 4.5",
            "valid china visa",
            "оформление паспорта",
            "оформить паспорт",
            "паспорт гражданина",
            "паспорт рк",
            "казахстанский паспорт",
            "фото на паспорт",
            "фотографии",
            "виза кнр",
        ),
    ):
        return DomainIntent(
            name="passport",
            allowed_sources=(SOURCE_PASSPORT,),
        )

    if _contains_any(
        q,
        (
            "consular fee",
            "processing time",
            "fees and processing",
            "консульский сбор",
            "срок оформления",
            "сроки оформления",
            "сколько времени",
            "стоимость",
            "оплата",
        ),
    ):
        return DomainIntent(
            name="fees_processing",
            allowed_sources=(SOURCE_FEES,),
        )

    if _contains_any(
        q,
        (
            "embassy",
            "consular section",
            "kazakhstan embassy in china",
            "посольство",
            "консульский отдел",
            "посольство казахстана в китае",
        ),
    ):
        return DomainIntent(
            name="overview",
            allowed_sources=(SOURCE_OVERVIEW,),
        )

    return DomainIntent(
        name="unknown",
        allowed_sources=(),
    )
