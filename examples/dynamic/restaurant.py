#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import os
import sys
from pathlib import Path
from typing import Dict, List

import aiohttp
from dotenv import load_dotenv
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.services.daily import DailyParams, DailyTransport

from pipecat_flows import FlowArgs, FlowConfig, FlowManager, FlowResult

sys.path.append(str(Path(__file__).parent.parent))
from runner import configure

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

# Flow Configuration - Food ordering with Cart
#
# This configuration defines a food ordering system with the following states:
#
# 1. start
#    - Initial state where user chooses between pizza or coke
#    - Functions:
#      * choose_pizza (transitions to choose_pizza)
#      * choose_coke (transitions to choose_coke)
#
# 2. choose_pizza
#    - Handles pizza order details
#    - Functions:
#      * select_pizza_order (node function with size and type)
#      * add_to_cart (transitions to add_to_cart)
#
# 3. choose_coke
#    - Handles coke order details
#    - Functions:
#      * select_coke_order (node function with size)
#      * add_to_cart (transitions to add_to_cart)
#
# 4. add_to_cart
#    - Adds the selected item to the cart
#    - Functions:
#      * continue_shopping (transitions to start)
#      * checkout (transitions to confirm)
#
# 5. confirm
#    - Reviews order details with the user
#    - Functions:
#      * complete_order (transitions to end)
#      * continue_shopping (transitions to start)
#
# 6. end
#    - Final state that closes the conversation
#    - No functions available
#    - Post-action: Ends conversation


# Type definitions
class PizzaOrderResult(FlowResult):
    size: str
    type: str
    price: float


class CokeOrderResult(FlowResult):
    size: str
    price: float


class CartItem:
    def __init__(self, item_type: str, details: dict, price: float):
        self.item_type = item_type
        self.details = details
        self.price = price


# Shopping cart to store items
class ShoppingCart:
    def __init__(self):
        self.items: List[CartItem] = []
        self.total_price: float = 0.0

    def add_item(self, item: CartItem):
        self.items.append(item)
        self.total_price += item.price

    def get_cart_summary(self) -> str:
        summary = "Cart Items:\n"
        for idx, item in enumerate(self.items, 1):
            if item.item_type == "pizza":
                summary += f"{idx}. {item.details['size'].capitalize()} {item.details['type'].capitalize()} Pizza - ${item.price:.2f}\n"
            elif item.item_type == "coke":
                summary += f"{idx}. {item.details['size'].capitalize()} Coke - ${item.price:.2f}\n"
        summary += f"\nTotal: ${self.total_price:.2f}"
        return summary


# Initialize shopping cart
shopping_cart = ShoppingCart()


# Function handlers
async def check_kitchen_status(action: dict) -> None:
    """Check if kitchen is open and log status."""
    logger.info("Checking kitchen status")


async def select_pizza_order(args: FlowArgs) -> PizzaOrderResult:
    """Handle pizza size and type selection."""
    size = args["size"]
    pizza_type = args["type"]

    # Simple pricing
    base_price = {"small": 10.00, "medium": 15.00, "large": 20.00}
    price = base_price[size]

    return {"size": size, "type": pizza_type, "price": price}


async def select_coke_order(args: FlowArgs) -> CokeOrderResult:
    """Handle coke size selection."""
    size = args["size"]

    # Simple pricing for coke
    base_price = {"small": 2.00, "medium": 3.00, "large": 4.00}
    price = base_price[size]

    return {"size": size, "price": price}


async def add_to_cart(args: FlowArgs) -> Dict:
    """Add the current item to the cart."""
    item_type = args["item_type"]
    item_details = args["item_details"]
    item_price = args["item_price"]
    
    cart_item = CartItem(item_type, item_details, item_price)
    shopping_cart.add_item(cart_item)
    
    return {"status": "added", "cart": shopping_cart.get_cart_summary()}


flow_config: FlowConfig = {
    "initial_node": "start",
    "nodes": {
        "start": {
            "role_messages": [
                {
                    "role": "system",
                    "content": "You are an order-taking assistant. You must ALWAYS use the available functions to progress the conversation. This is a phone conversation and your responses will be converted to audio. Keep the conversation friendly, casual, and polite. Avoid outputting special characters and emojis.",
                }
            ],
            "task_messages": [
                {
                    "role": "system",
                    "content": "For this step, ask the user if they want pizza or coke, and wait for them to use a function to choose. Start off by greeting them if this is the first interaction. Be friendly and casual; you're taking a food order over the phone. If they're returning to order more, remind them of their current cart items.",
                }
            ],
            "pre_actions": [
                {
                    "type": "check_kitchen",
                    "handler": check_kitchen_status,
                },
            ],
            "functions": [
                {
                    "type": "function",
                    "function": {
                        "name": "choose_pizza",
                        "description": "User wants to order pizza. Let's get that order started.",
                        "parameters": {"type": "object", "properties": {}},
                        "transition_to": "choose_pizza",
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "choose_coke",
                        "description": "User wants to order coke. Let's get that order started.",
                        "parameters": {"type": "object", "properties": {}},
                        "transition_to": "choose_coke",
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "checkout",
                        "description": "User is ready to checkout with current items in cart.",
                        "parameters": {"type": "object", "properties": {}},
                        "transition_to": "confirm",
                    },
                },
            ],
        },
        "choose_pizza": {
            "task_messages": [
                {
                    "role": "system",
                    "content": """You are handling a pizza order. Use the available functions:
- Use select_pizza_order when the user specifies both size AND type

Pricing:
- Small: $10
- Medium: $15
- Large: $20

Remember to be friendly and casual.""",
                }
            ],
            "functions": [
                {
                    "type": "function",
                    "function": {
                        "name": "select_pizza_order",
                        "handler": select_pizza_order,
                        "description": "Record the pizza order details",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "size": {
                                    "type": "string",
                                    "enum": ["small", "medium", "large"],
                                    "description": "Size of the pizza",
                                },
                                "type": {
                                    "type": "string",
                                    "enum": ["pepperoni", "cheese", "supreme", "vegetarian"],
                                    "description": "Type of pizza",
                                },
                            },
                            "required": ["size", "type"],
                        },
                        "transition_to": "add_to_cart",
                    },
                },
            ],
        },
        "choose_coke": {
            "task_messages": [
                {
                    "role": "system",
                    "content": """You are handling a coke order. Use the available functions:
- Use select_coke_order when the user specifies the size

Pricing:
- Small: $2
- Medium: $3
- Large: $4

Remember to be friendly and casual.""",
                }
            ],
            "functions": [
                {
                    "type": "function",
                    "function": {
                        "name": "select_coke_order",
                        "handler": select_coke_order,
                        "description": "Record the coke order details",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "size": {
                                    "type": "string",
                                    "enum": ["small", "medium", "large"],
                                    "description": "Size of the coke",
                                },
                            },
                            "required": ["size"],
                        },
                        "transition_to": "add_to_cart",
                    },
                },
            ],
        },
        "add_to_cart": {
            "task_messages": [
                {
                    "role": "system",
                    "content": """The user has selected an item. Confirm that the item has been added to their cart and ask if they want to continue shopping or check out.
                    
Use the available functions:
- Use add_to_cart to add the current item to the cart
- The user can then either continue_shopping or checkout

Be friendly and clear when confirming the item has been added to the cart.""",
                }
            ],
            "functions": [
                {
                    "type": "function",
                    "function": {
                        "name": "add_to_cart",
                        "handler": add_to_cart,
                        "description": "Add the current item to the shopping cart",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "item_type": {
                                    "type": "string",
                                    "enum": ["pizza", "coke"],
                                    "description": "Type of item being added to cart",
                                },
                                "item_details": {
                                    "type": "object",
                                    "description": "Details of the item being added",
                                },
                                "item_price": {
                                    "type": "number",
                                    "description": "Price of the item being added",
                                },
                            },
                            "required": ["item_type", "item_details", "item_price"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "continue_shopping",
                        "description": "User wants to add more items to their order",
                        "parameters": {"type": "object", "properties": {}},
                        "transition_to": "start",
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "checkout",
                        "description": "User is ready to checkout with current items in cart",
                        "parameters": {"type": "object", "properties": {}},
                        "transition_to": "confirm",
                    },
                },
            ],
        },
        "confirm": {
            "task_messages": [
                {
                    "role": "system",
                    "content": """Read back the complete order details to the user from the cart and ask if they want to complete the order or continue shopping. Use the available functions:
- Use complete_order when the user confirms that the order is correct
- Use continue_shopping if they want to add more items

Be friendly and clear when reading back the order details.""",
                }
            ],
            "functions": [
                {
                    "type": "function",
                    "function": {
                        "name": "complete_order",
                        "description": "User confirms the order is correct",
                        "parameters": {"type": "object", "properties": {}},
                        "transition_to": "end",
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "continue_shopping",
                        "description": "User wants to add more items to their order",
                        "parameters": {"type": "object", "properties": {}},
                        "transition_to": "start",
                    },
                },
            ],
        },
        "end": {
            "task_messages": [
                {
                    "role": "system",
                    "content": "Thank the user for their order and end the conversation politely and concisely. Mention that their order will be ready soon.",
                }
            ],
            "functions": [],
            "post_actions": [{"type": "end_conversation"}],
        },
    },
}


async def main():
    """Main function to set up and run the food ordering bot."""
    async with aiohttp.ClientSession() as session:
        (room_url, _) = await configure(session)

        # Initialize services
        transport = DailyTransport(
            room_url,
            None,
            "Food Ordering Bot",
            DailyParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                vad_analyzer=SileroVADAnalyzer(),
            ),
        )

        stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))
        tts = CartesiaTTSService(
            api_key=os.getenv("CARTESIA_API_KEY"),
            voice_id="820a3788-2b37-4d21-847a-b65d8a68c99a",  # Salesman
        )
        llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o")

        context = OpenAILLMContext()
        context_aggregator = llm.create_context_aggregator(context)

        # Create pipeline
        pipeline = Pipeline(
            [
                transport.input(),
                stt,
                context_aggregator.user(),
                llm,
                tts,
                transport.output(),
                context_aggregator.assistant(),
            ]
        )

        task = PipelineTask(pipeline, params=PipelineParams(allow_interruptions=True))

        # Initialize flow manager in static mode
        flow_manager = FlowManager(
            task=task,
            llm=llm,
            context_aggregator=context_aggregator,
            tts=tts,
            flow_config=flow_config,
        )

        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            await transport.capture_participant_transcription(participant["id"])
            logger.debug("Initializing flow")
            await flow_manager.initialize()

        runner = PipelineRunner()
        await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
