from openai import OpenAI

def askLLM(prompt, mymodel):

    if mymodel == "gpt-4o" or mymodel == "gpt-5":
        client = OpenAI(
            api_key="ABCD"
        )
    else:
        client = OpenAI(
            api_key="ABCD",
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )

    response = client.chat.completions.create(
        model=mymodel,
        messages=[
            {
                "role": "system",
                "content": "You are an expert in analog circuit design.",
            },
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content




def getMyRefine(queryId,topology,topologyId,sizingValues):
    import time
    currentTime = time.time()
    logFilePath = f"logRefinequeryid{queryId}_topologyId{topologyId}_{currentTime}.log"
    log = open(logFilePath, "w")
    with open(f"../tasks/{queryId}.prompt", "r") as f:
        spec = f.read()
    iteration = 0
    current_best = ""
    history_design = []
    MAX_ITERATION = 3
    iter = f"""
In the previous iteration, I have the original topology which reaches the following specifications under the best sizing: 
{sizingValues}
"""
    while iteration <=MAX_ITERATION:
        iteration += 1
        if history_design:
            iter += "In the previous iterations, I have the following design and the results."
            for i, h in enumerate(history_design):
                iter += f"\n Design {i+1}: {h}"
            iter += "\n Now please try to refine the topology again. \n"
        prompt=iter + f"""
Please design an operational amplifier using the skywater PDK (SKY130 process node) that meets the following specifications:
{spec}
Now I have the following topology in netlist format:
{topology}
In this step, you would need to judge whether it is needed to do a topology refinement.

There is a knowledge agent that can help to find relevant knowledge documents about analog circuit design from a knowledge base.
You can write a query to ask the knowledge agent to get relevant knowledge documents to help you design the op-amp.
To ask the agent, please first analyze the specifications and then output the query text in a code block. 
You do not need to do detail refinement now, just need to find relevant knowledge documents.

Example
```text
How to optimize the topology of a two-stage operational amplifier with Miller compensation?
```
"""
        print("First query:")
        log.write("First query:\n" + prompt + "\n")
        print(prompt)
        first_query = askLLM(prompt, "gemini-2.5-flash")
        print("First query result:")
        print(first_query)
        log.write("First query result:\n" + first_query + "\n")

        import knowledge
        query_text_list = knowledge.extract_code(first_query)
        print("Extracted query text list:")
        print(query_text_list)
        log.write("Extracted query text list:\n" + str(query_text_list) + "\n")
        knowledge_results = knowledge.getKnowledge(query_text_list, mytop_k=5)
        prompt = iter+f"""
Please design an operational amplifier using the skywater PDK (SKY130 process node) that meets the following specifications:
{spec}
Now I have the following topology in netlist format:
{topology}
In this step, you would need to judge whether it is needed to do a topology refinement.

The following relevant knowledge documents are retrieved to help to design the op-amp:
{knowledge_results}
Please *refine and modify the topology if needed (You could do small changes on the topology like adding Gain Boosting or compensation or other small structures without changing the main structure. Do not do large changes like changing from one stage to two stages)* and then convert it into a PySpice format code like the example below.

Example format(Do not use advanced functions like self.parameter, just write one device with sizing variables in plain strings one by one):
```python
class OpAmp(SubCircuitFactory):
    __name__ = "op_amp"
    # The external nodes of the subcircuit, matching the .subckt line in SPICE
    __nodes__ = ("gnda", "vdda", "vinn", "vinp", "vout")

    def __init__(self):
        SubCircuit.__init__(self, self.__name__, *self.__nodes__)

        # MOSFETs (xm<id> <drain> <gate> <source> <bulk> <model_name> l='L' w='W' m='M')
        # PySpice: self.X("<id>", "<model_name>", drain_node, gate_node, source_node, bulk_node, l=..., w=..., m=...)

        # PFET devices
        self.X(
            "M8", "sky130_fd_pr__pfet_01v8", "net36", "vb2", "vdda", "vdda",
            l='P1_L', w='P1_W*1', m='P1_M'
        )
        self.X(
            "M7", "sky130_fd_pr__pfet_01v8", "net30", "vb1", "vdda", "vdda",
            l='P2_L', w='P2_W*1', m='P2_M'
        )
        self.X(
            "M5", "sky130_fd_pr__pfet_01v8", "vout", "vb4", "vdda", "vdda",
            l='P3_L', w='P3_W*1', m='P3_M'
        )

        # NFET devices
        self.X(
            "M6", "sky130_fd_pr__nfet_01v8", "net33", "vb3", "gnda", "gnda",
            l='N1_L', w='N1_W*1', m='N1_M'
        )
        self.X(
            "M1", "sky130_fd_pr__nfet_01v8", "net36", "vinn", "net33", "gnda",
            l='N2_L', w='N2_W*1', m='N2_M'
        )
        self.X(
            "M0", "sky130_fd_pr__nfet_01v8", "net30", "vinp", "net33", "gnda",
            l='N3_L', w='N3_W*1', m='N3_M'
        )
        self.X(
            "M4", "sky130_fd_pr__nfet_01v8", "vout", "net36", "gnda", "gnda",
            l='N4_L', w='N4_W*1', m='N4_M'
        )
        self.X(
            "M3", "sky130_fd_pr__nfet_01v8", "net36", "net30", "gnda", "gnda",
            l='N5_L', w='N5_W*1', m='N5_M'
        )
        self.X(
            "M2", "sky130_fd_pr__nfet_01v8", "net30", "net30", "gnda", "gnda",
            l='N6_L', w='N6_W*1', m='N6_M'
        )

        # Capacitor (C<id> <node1> <node2> <value>)
        # PySpice: self.C(<id>, node1, node2, value)
        self.C(0, "vout", "net36", 'CAPACITOR_0')

        # Bias Voltages (V<id> <positive_node> <negative_node> <value>)
        # PySpice: self.V(<id>, positive_node, negative_node, value)
        # The '0' node in SPICE typically refers to the global ground,
        # which maps to the 'gnda' node of this subcircuit.
        self.V(1, "vb1", "gnda", 'VB1')
        self.V(2, "vb2", "gnda", 'VB2')
        self.V(3, "vb3", "gnda", 'VB3')
        self.V(4, "vb4", "gnda", 'VB4')
```
Please also *add the bias voltage* in this class.
This class will be used to be loaded into a fixed test bench later, so please do not change any signature of the class and the nodes so that it can be loaded correctly.
The circuit name is "op_amp". Do not change it.
For the parameter sizing, please keep the **original variables** in the netlist and also include the bias voltage in the class.
"""
        history = prompt
        print("Second query:")
        print(prompt)
        log.write("Second query:\n" + prompt + "\n")
        second_query = askLLM(prompt, "gemini-2.5-flash")
        print("Second query result:")
        print(second_query)
        log.write("Second query result:\n" + second_query + "\n")
    return answer



if __name__ == "__main__":
    with open("", "r") as f:
        topology = f.read()
    getMyRefine(1, topology,topologyId=37)