
# real-time.pyより

# # 一旦削除
# # 静的パラメータをコンソールから入力（改善提案2）
# def input_static_parameters(static_param_control):
#     static_params = {}
#     param_names = ['Age', 'Sex', 'Length', 'ShoeSize', 'StandNum',
#                    'Risk C', 'Risk D', 'Risk E', 'Risk F', 'Risk G', 'Risk H', 'Risk I']
#     for idx, param_name in enumerate(param_names):
#         if static_param_control[idx]:
#             if param_name == 'Sex':
#                 value = input(f"入力してください {param_name}  ('M' or 'F'): ")
#                 value = 0 if value == 'M' else 1
#             else:
#                 value = float(input(f"入力してください {param_name} : "))
#             static_params[param_name] = value
#     return static_params

# # 一旦削除
# # 静的パラメータをテンソルに変換（改善提案2）
# def process_static_params(static_params, static_param_control):
#     param_names = ['Age', 'Sex', 'Length', 'ShoeSize', 'StandNum',
#                    'Risk C', 'Risk D', 'Risk E', 'Risk F', 'Risk G', 'Risk H', 'Risk I']
#     selected_params = [static_params[name] for idx, name in enumerate(param_names) if static_param_control[idx]]
#     # すべての値を float 型に統一
#     selected_params = [float(value) for value in selected_params]
#     return torch.tensor(selected_params, dtype=torch.float32).to(device)


# 静的パラメータを入力
# static_params = input_static_parameters(static_param_control)
# static_params_tensor = process_static_params(static_params, static_param_control)

# 静的パラメータの入力次元を取得（改善提案4）
# static_input_dim = static_params_tensor.shape[0]
##### → model = create_model(input_dim, static_input_dim, class_num)