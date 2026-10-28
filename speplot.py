import matplotlib.pyplot as plt

# Кількість воркерів
workers = [1, 2, 4, 8, 12]

# Прискорення (Speedup) з вашої таблиці
speedup_threads_gil = [1.0, 0.93, 1.03, 1.07, 0.94]
speedup_processes = [1.0, 1.30, 1.91, 1.93, 1.87]
speedup_threads_nogil = [1.0, 1.45, 1.89, 1.99, 2.09]

plt.figure(figsize=(10, 6))

# Ідеальне лінійне прискорення
plt.plot(workers, workers, 'k--', label='Ідеальне прискорення (Linear)')

# Дані бенчмарків
plt.plot(workers, speedup_threads_gil, marker='o', label='Threads (GIL увімкнено)')
plt.plot(workers, speedup_processes, marker='s', label='Processes')
plt.plot(workers, speedup_threads_nogil, marker='^', label='Threads (3.13t без GIL)')

plt.title('Графік прискорення (Speedup) залежно від кількості воркерів')
plt.xlabel('Кількість воркерів (N)')
plt.ylabel('Прискорення (Speedup)')
plt.xticks(workers)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

# Збереження графіка
plt.savefig('speedup.png', dpi=300, bbox_inches='tight')
print("Графік успішно збережено у файл speedup.png")
