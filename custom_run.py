

from src.conversation import Conversation, SeparatorStyle
from src.model import GPTQModel


default_template = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. "
           "The assistant gives helpful, detailed, and polite answers to the human's questions.",
    roles=["Human", "Assistant"],
    messages=[
        ["Human", "Give three tips for staying healthy."],
        ["Assistant",
            "Sure, here are three tips for staying healthy:\n"
            "1. Exercise regularly: Regular physical activity can help improve your overall health and wellbeing. "
            "It can also help reduce your risk of chronic conditions such as obesity, diabetes, heart disease, "
            "and certain cancers. Aim for at least 150 minutes of moderate-intensity aerobic exercise or "
            "75 minutes of vigorous-intensity aerobic exercise per week, along with muscle-strengthening "
            "activities at least two days per week.\n"
            "2. Eat a balanced diet: Eating a balanced diet that is rich in fruits, "
            "vegetables, whole grains, lean proteins, and healthy fats can help support "
            "your overall health. Try to limit your intake of processed and high-sugar foods, "
            "and aim to drink plenty of water throughout the day.\n"
            "3. Get enough sleep: Getting enough quality sleep is essential for your physical "
            "and mental health. Adults should aim for seven to nine hours of sleep per night. "
            "Establish a regular sleep schedule and try to create a relaxing bedtime routine to "
            "help improve the quality of your sleep."]
    ],
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)

context_template = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant."
           "The assistant gives helpful, detailed, and polite answers to the human's questions, based only on the context provided by the human."
           "The context is represeted by the tag <context> and end on </context>.",
    roles=["Human", "Assistant"],
    messages=[
        ["Human", "<context>The monkeys like bananas</context> Please tell my what monkeys like."],
        ["Assistant", "Sure, monkeys like bananas"]
    ],
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)


model = GPTQModel(model_name="anon8231489123/vicuna-13b-GPTQ-4bit-128g", device="cuda", wbits=4, groupsize=128)
my_conv = context_template.copy()
my_conv = model.inference({"prompt": "Fred likes to to travel and go to the GYM, Please tell my what fred do.",
                        "conversation": my_conv})

print(my_conv.last_message)