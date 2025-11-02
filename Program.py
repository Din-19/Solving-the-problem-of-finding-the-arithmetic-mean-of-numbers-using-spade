import asyncio
import random
from spade.agent import Agent
from spade.message import Message
import networkx as nx
import matplotlib.pyplot as plt

NUM_AGENTS = 10
TARGET_PRECISION = 1e-4
MAX_ITER = 50

message_boxes = {f"agent{i}@localhost": [] for i in range(NUM_AGENTS)}


class ConsensusAgent(Agent):
    def __init__(self, jid, password, initial_value, neighbours):
        super().__init__(jid, password)
        self.value = float(initial_value)
        self.neighbours = neighbours

    async def send(self, msg):
        recipient = str(msg.to).split("/")[0]
        if recipient in message_boxes:
            message_boxes[recipient].append(msg)

    async def receive(self, timeout=None):
        my_jid = str(self.jid).split("/")[0]
        if message_boxes[my_jid]:
            return message_boxes[my_jid].pop(0)
        return None


async def main():
    print("Генерируем начальные значения...")
    initial_values = [random.uniform(0, 100) for _ in range(NUM_AGENTS)]
    true_mean = sum(initial_values) / NUM_AGENTS
    print(f"Истинное среднее: {true_mean:.6f}\n")

    print("Строим топологию агентов...")
    G = nx.gnm_random_graph(NUM_AGENTS, 15)
    while not nx.is_connected(G):
        G = nx.gnm_random_graph(NUM_AGENTS, 15)

    print("Открываем схему агентов. Закройте окно, чтобы продолжить...\n")
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42)

    labels = {
        i: f"agent{i}\n{initial_values[i]:.2f}"
        for i in range(NUM_AGENTS)
    }

    nx.draw(
        G,
        pos,
        labels=labels,
        with_labels=True,
        node_color="#6a5acd",
        node_size=2000,
        font_size=9,
        font_color="white",
        font_weight="bold",
        edge_color="#555555",
        width=1.5
    )
    plt.title("Топология агентов и их загаданные числа", fontsize=14, pad=20)
    plt.tight_layout()
    plt.show()

    agents = []
    for i in range(NUM_AGENTS):
        jid = f"agent{i}@localhost"
        neighbours = [f"agent{j}@localhost" for j in G.neighbors(i)]
        agent = ConsensusAgent(jid, "secret", initial_values[i], neighbours)
        agents.append(agent)

    print("Агенты созданы. Запускаем вычисление общего среднего...\n")

    current_values = {f"agent{i}@localhost": initial_values[i] for i in range(NUM_AGENTS)}
    neighbours_map = {
        f"agent{i}@localhost": [f"agent{j}@localhost" for j in G.neighbors(i)]
        for i in range(NUM_AGENTS)
    }

    for iteration in range(MAX_ITER):
        for box in message_boxes.values():
            box.clear()

        for jid in current_values:
            for nb in neighbours_map[jid]:
                msg = Message(to=nb)
                msg.body = str(current_values[jid])
                sender = next(a for a in agents if str(a.jid).split("/")[0] == jid)
                await sender.send(msg)

        new_values = {}
        max_diff = 0.0
        for jid in current_values:
            received = []
            agent = next(a for a in agents if str(a.jid).split("/")[0] == jid)
            while True:
                msg = await agent.receive()
                if msg is None:
                    break
                try:
                    received.append(float(msg.body))
                except ValueError:
                    continue

            all_vals = [current_values[jid]] + received
            new_val = sum(all_vals) / len(all_vals)
            new_values[jid] = new_val
            max_diff = max(max_diff, abs(new_val - current_values[jid]))

        current_values = new_values

        if max_diff < TARGET_PRECISION:
            print(f"Общее среднее найдено на итерации {iteration + 1}!")
            break
    else:
        print("Достигнут лимит итераций — возможно, сеть слишком разрежена.")

    final_val = list(current_values.values())[0]
    print(f"\nРезультат: {final_val:.6f}")
    print(f"Истинное среднее:     {true_mean:.6f}")
    print(f"Ошибка:               {abs(final_val - true_mean):.2e}")


if __name__ == "__main__":
    asyncio.run(main())