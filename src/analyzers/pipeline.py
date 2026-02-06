from src.analyzers.sentiment import SentimentAnalyzer
from src.analyzers.summarizer import Summarizer
from src.analyzers.topic_extractor import TopicExtractor
from src.clients.llm_factory import LLMClient
from src.storage.repository import Repository
from src.storage.models import CollectedItemDB
from src.logging_config import get_logger

logger = get_logger(__name__)


class AnalysisPipeline:
    """Runs the full analysis pipeline on collected items."""

    def __init__(self, llm: LLMClient, repo: Repository):
        self.sentiment = SentimentAnalyzer(llm)
        self.summarizer = Summarizer(llm)
        self.topics = TopicExtractor(llm)
        self.repo = repo
        self.model_name = getattr(llm, "default_model", "unknown")

    async def analyze_item(self, item: CollectedItemDB) -> dict:
        text = f"{item.title}\n\n{item.content}"
        logger.info("analyzing_item", item_id=item.id, title=item.title[:50])

        try:
            # Run all analyses
            sentiment_result = await self.sentiment.analyze(text)
            topic_result = await self.topics.extract(text)
            summary = await self.summarizer.summarize(text)

            # Save sentiment analysis
            await self.repo.save_analysis(
                item_id=item.id,
                analysis_type="sentiment",
                result_text=sentiment_result.get("reasoning", ""),
                sentiment=sentiment_result.get("sentiment", "neutral"),
                confidence=sentiment_result.get("confidence", 0.0),
                model_used=self.model_name,
            )

            # Save topic extraction
            await self.repo.save_analysis(
                item_id=item.id,
                analysis_type="topics",
                result_text=topic_result.get("primary_topic", ""),
                topics=topic_result.get("topics", []),
                model_used=self.model_name,
            )

            # Save summary
            await self.repo.save_analysis(
                item_id=item.id,
                analysis_type="summary",
                result_text=summary,
                model_used=self.model_name,
            )

            await self.repo.mark_item_analyzed(item.id, "completed")
            logger.info("item_analyzed", item_id=item.id)

            return {
                "item_id": item.id,
                "sentiment": sentiment_result,
                "topics": topic_result,
                "summary": summary,
            }

        except Exception as e:
            logger.error("analysis_failed", item_id=item.id, error=str(e))
            await self.repo.mark_item_analyzed(item.id, "failed")
            raise

    async def process_pending(self, limit: int = 50) -> list[dict]:
        items = await self.repo.get_pending_items(limit=limit)
        logger.info("processing_pending", count=len(items))

        results = []
        for item in items:
            try:
                result = await self.analyze_item(item)
                results.append(result)
            except Exception:
                continue

        return results
