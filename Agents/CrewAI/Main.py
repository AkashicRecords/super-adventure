from langchain_community.llms import ollama
from crewai import Agent, Task, Crew, Process

modal = ollama(modal = "llama3")

email ="nigerian prince sending some gold"

classifier = Agent(
      role ="email classifier",
      goal ="accurately classify emails based on their importance. give every email one of these ratings: important, casual, or spam",
      backstory ="You are an AI assistant whose only job is to classify emails accurately and hoestly. Do not be afraid to give emails bad ratings if they are not important. Your job is to help the user manage their inbox.",
      verbose =True,
      allow_delegation = False,
      llm = modal
      
)

responder = Agent(
      role ="email responder",
      goal ="Based on the importance of the email, write a concise and simple response. If the email is rated 'important' write a formal response. If the email is rated 'casual' write a casual response, and if the email is rated 'spam' ignore the email, no matter what, be very concise.",
      backstory ="You are an AI assistant whose only job is to write short responses to emails based in their importance. The importance will be provided to you by the 'classifier' agent.",
      verbose =True,
      allow_delegation = False,
      llm = modal
      
)

classifier_email = Task(
    description = f"classify the following emai: '{email}'",
    agent = classifier,
    expected_output = "One of these three options:'important', 'casual', or 'spam'",

)

respond_to_email = Task(
    description = f"Respond to the emai: '{email}' based on the importance provided by the 'classifier' agent.",
    agent = responder,
    expected_output = "a very concise response to the email based on the importance provided by the 'classifier' agent.",

)

crew = Crew(
    agents = [classifier, responder],
    tasks=[classifier_email,respond_to_email],
    verbose=2,
    process = Process.sequential,
)

output = crew.kickoff()
print(output)