"""Scripted model boundary: task proposals accompany the first native action."""

from ai_service.capabilities.assistant.react.actions import RespondAction


class WithTaskProposals:
    def __init__(self, model, proposals):
        self.model, self.proposals = model, proposals

    async def decide(self, message, transcript, allowed, repair, context=None):
        decision = await self.model.decide(
            message, transcript, allowed, repair, context
        )
        if (
            decision is not None
            and context is not None
            and context.task_registration_open
            and decision.registration is None
            and not isinstance(decision.action, RespondAction)
        ):
            draft = await self.proposals.extract_tasks(message)
            if draft.tasks:
                decision = decision.model_copy(update={"registration": draft})
        return decision
