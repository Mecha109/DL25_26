import os
import re
from collections import defaultdict
import matplotlib.pyplot as plt

RESULTS_FILE = "table_results.txt"
OUTPUT_FIG = "best_train_acc_vs_hidden.png"

def parse_results(path: str):
	"""Parse the results table and return list of dict rows."""
	rows = []
	if not os.path.isfile(path):
		raise FileNotFoundError(f"Results file not found: {path}")
	with open(path, "r", encoding="utf-8") as f:
		for line in f:
			line = line.strip()
			# Skip separators and headers
			if not line or line.startswith("=") or line.startswith("-") or line.startswith("Hidden"):
				continue
			# Expect lines starting with an integer hidden size
			if not re.match(r"^\d+\s+", line):
				continue
			parts = re.split(r"\s+", line)
			if len(parts) < 7:
				continue
			try:
				hidden = int(parts[0])
				lr = float(parts[1])
				dropout = float(parts[2])
				l2 = float(parts[3])
				train_acc = float(parts[4])
				val_acc = float(parts[5])
				test_acc = float(parts[6])
			except ValueError:
				continue
			rows.append({
				"hidden": hidden,
				"lr": lr,
				"dropout": dropout,
				"l2": l2,
				"train_acc": train_acc,
				"val_acc": val_acc,
				"test_acc": test_acc,
			})
	return rows

def best_train_acc_by_hidden(rows):
	best = defaultdict(lambda: -1.0)
	for r in rows:
		h = r["hidden"]
		if r["train_acc"] > best[h]:
			best[h] = r["train_acc"]
	# Return sorted by hidden size
	return dict(sorted(best.items(), key=lambda kv: kv[0]))

def plot_best_train_acc(best_map):
	hidden_sizes = list(best_map.keys())
	best_accs = [best_map[h] for h in hidden_sizes]
	plt.figure(figsize=(6,4))
	plt.plot(hidden_sizes, best_accs, marker='o')
	plt.title('Best Training Accuracy vs Hidden Layer Size')
	plt.xlabel('Hidden Layer Size')
	plt.ylabel('Best Training Accuracy')
	plt.grid(True, alpha=0.3)
	plt.tight_layout()
	plt.savefig(OUTPUT_FIG, dpi=150)
	return OUTPUT_FIG

def main():
	rows = parse_results(RESULTS_FILE)
	best_map = best_train_acc_by_hidden(rows)
	fig_path = plot_best_train_acc(best_map)
	print("Best training accuracy per hidden size:")
	for h, acc in best_map.items():
		print(f"Hidden {h}: {acc:.4f}")
	print(f"Plot saved to {fig_path}")

if __name__ == "__main__":
	main()

