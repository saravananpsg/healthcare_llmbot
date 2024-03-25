from langchain.llms import OpenAI
import openai
from langchain.evaluation.qa import QAEvalChain
from langchain.chains import RetrievalQA
from read_data import get_vector_store
import dotenv
import uuid
import os
from utils import write_json

config = dotenv.dotenv_values(".env")
openai.api_key = config['OPENAI_API_KEY']
openai_api_key = config['OPENAI_API_KEY']

vector_store, df = get_vector_store()


ground_truth_question_answers = [
    {'question': "What is gestational diabetes and how is it diagnosed?",
     'answer': 'Gestational diabetes is a type of diabetes that develops during pregnancy and usually goes away after delivery. It is diagnosed using a 3-point 75 g oral glucose tolerance test (OGTT) at 24 to 28 weeks of gestation, unless the woman has already been diagnosed with diabetes or pre-diabetes. It is important to also screen for pre-existing diabetes in the first trimester and after delivery, as women with a history of GDM are at increased risk of developing type 2 diabetes later on in life.'
     },
    {
        'question': "What are some healthy eating tips for people with diabetes?",
        'answer': "Some healthy eating tips for people with diabetes include:\n\n1. Focus on a balanced diet that includes carbohydrates, protein, and fats, with an emphasis on managing carbohydrate intake to control blood sugar levels.\n2. Choose healthier cooking methods like steaming, baking, boiling, or grilling to prepare meals.\n3. Opt for whole grains over refined grains, such as replacing white rice with brown rice.\n4. Select lean meats and remove visible fats before cooking to reduce saturated fat intake.\n5. Use natural seasonings like herbs and spices instead of excessive salt.\n6. Incorporate vegetables and fruits as the main components of your meals, making up at least 50% of your plate.\n7. Stay hydrated with water as your primary drink choice and avoid sugary beverages.\n8. Plan meals ahead, make a shopping list, and opt for healthier products during festivals and celebrations to maintain healthy eating habits.\n9. Communicate your boundaries politely when faced with peer pressure to indulge in unhealthy foods during social gatherings.\n\nRemember, personalized nutritional advice from a healthcare professional, such as a dietitian, can further enhance your diabetes management through tailored dietary recommendations."
    },
    {
        'question': "How can my outpatient bill for diabetes be covered?",
        'answer': "Your outpatient bill for diabetes can be covered through various means, including government subsidies, employee benefits/private medical insurance, and the use of MediSave through the Chronic Disease Management Programme (CDMP). The bill can be further offset with government subsidies available at public specialist outpatient clinics, polyclinics, and through schemes like the Community Health Assist Scheme (CHAS), Pioneer Generation (PG), and Merdeka Generation (MG) outpatient subsidies. Additionally, patients can tap on accounts of immediate family members for MediSave, and those aged 60 and above can use MediSave for the 15% co-payment under CDMP."
    }
]


if __name__ == "__main__":

    llm = OpenAI(temperature=0.1, openai_api_key=openai_api_key)
    chain = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type="stuff",
                                        retriever=vector_store.as_retriever(),
                                        input_key="question")
    predictions = chain.apply(ground_truth_question_answers)
    api_filename = "retrieval_qa_" + str(uuid.uuid4())
    api_resp_file = os.path.join("results", api_filename)
    write_json(api_resp_file, predictions)

    # Start your eval chain
    eval_chain = QAEvalChain.from_llm(llm)
    eval_outputs = eval_chain.evaluate(ground_truth_question_answers,
                                       predictions,
                                       question_key="question",
                                       prediction_key="result",
                                       answer_key='answer')
    print(eval_outputs)
