import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from scipy.optimize import minimize, Bounds


class BaseDecomposer:
    def __init__(self,
                 data: pd.DataFrame,
                 branches: list,
                 n_components: int = 2):
        """
        :param data: таблица pandas с данными, где по строкам время, а по столбцам отрасли
        :param branches: названия колонок из таблицы, которые надо декомпозировать
        :param n_components: количество компонент
        """
        self.data = data
        self.branches = branches
        for branch in branches:
            assert branch in data.columns, f"Отрасли {branch} нет в таблице с данными."
        self.n_branches = len(branches)
        assert n_components >= 1, "Число компонент должно быть >= 1"
        assert n_components == int(n_components), "Число компонент должно быть целым"
        self.n_components = n_components
        self.components = np.zeros((self.n_components, data.shape[0]))
        self.base_coefficients = np.zeros((self.n_components + 1))
        self.targets = data[branches]
        self.__targets = self.targets.to_numpy().T
        self.predictions = pd.DataFrame(np.zeros_like(self.targets),
                                        columns=self.targets.columns,
                                        index=self.targets.index)
        self.rho = np.zeros((self.n_branches, 1))
        self._alpha = np.zeros((self.n_branches, n_components))
        self.rng = np.random.default_rng(0)

        self.upper_bound = None
        self.lower_bound = None

        self.make_lower_bound()
        self.make_upper_bound()

    @property
    def alpha(self):
        """
        :return: L-1 нормализованные коэффициенты alpha
        """
        return self._alpha / self._alpha.sum(axis=1).reshape(-1, 1)

    def estimate_prices(self,
                        rho: np.ndarray,
                        alpha: np.ndarray):
        """
        Оценить цены для всех отраслей исходя из коэффициентов rho и alpha
        """
        power = rho / (rho - 1)
        prices = (self.components / self.base_coefficients[1:].reshape(-1, 1))
        prices = prices.reshape(*prices.shape, 1) ** power.T
        alphas = alpha.T.reshape(self.n_components, 1, self.n_branches)
        return self.base_coefficients[0] * (alphas * prices).sum(axis=0).T ** (1 / power)

    def make_lower_bound(self, alpha_lb=0, rho_lb=1, coeffs_lb=1e-3, components_lb=1e-3, eps=1e-6):
        """
        Изменить нижние границы параметров модели
        """
        self.lower_bound = np.concatenate((
            np.full_like(self._alpha, alpha_lb).flatten(),
            np.full_like(self.rho, rho_lb).flatten(),
            np.full_like(self.base_coefficients, coeffs_lb),
            np.full_like(self.components, components_lb).flatten())) + eps
        return self

    def make_upper_bound(self, alpha_ub=1, rho_ub=10, coeffs_ub=2.5, components_ub=3, eps=1e-6):
        """
        Изменить верхние границы параметров модели
        """
        self.upper_bound = np.concatenate((
            np.full_like(self._alpha, alpha_ub).flatten(),
            np.full_like(self.rho, rho_ub).flatten(),
            np.full_like(self.base_coefficients, coeffs_ub),
            np.full_like(self.components, components_ub).flatten())) - eps
        return self

    def fit(self, multistarts=1, reg=0., x0=None, verbose=False, mode='scipy', loss_function='mse', *args, **kwargs):
        """
        Обучить модель
        :param multistarts:
        :param reg:
        :param x0:
        :param verbose: выводить ли отладку из scipy
        :param mode:
        :param loss_function: ['mse', 'mre', 'mae']
        :param args: аргументы для scipy-minimize
        :param kwargs: ключевые аргументы для scipy-minimize
        :return:
        """
        if mode == 'scipy':
            self.fit_scipy(multistarts, reg, x0, verbose, loss_function, *args, **kwargs)
        else:
            raise NotImplementedError
        return self

    def calc_loss(self, reg=0., loss_function='mse'):
        """
        :param reg: коэффициент регуляризации за негладкость
        :param loss_function: функция потерь: mre, mse или mae
        :return:
        """
        self.predictions = self.estimate_prices(self.rho, self.alpha)
        if loss_function == 'mre':
            total_loss = ((1 - self.predictions / self.__targets) ** 2).sum()
        elif loss_function == 'mse':
            total_loss = np.sum((self.__targets - self.predictions) ** 2)
        else:
            total_loss = np.sum(np.abs(self.__targets - self.predictions))
        if reg != 0:
            diff = self.components[:, 1:] - self.components[:, :-1]
            total_loss += np.sum(diff[:, 1:] ** 2 + diff[:, :-1] ** 2) * reg
        return total_loss

    def fit_scipy(self, multistarts=1, reg=0., x0=None, verbose=False, loss_function='mse', *args, **kwargs):
        idx = np.cumsum([self._alpha.size, self.rho.size, self.base_coefficients.size])

        def target_function(params):
            alpha, rho, base_coefficients, components = np.split(params, idx)
            self._alpha = alpha.reshape(self._alpha.shape)
            self.rho = rho.reshape(self.rho.shape)
            self.base_coefficients = base_coefficients
            self.components = components.reshape(self.components.shape)
            return self.calc_loss(reg, loss_function=loss_function)

        if x0 is None:
            initial = lambda: self.rng.uniform(self.lower_bound, self.upper_bound)
        elif isinstance(x0, np.ndarray):
            initial = lambda: x0
        else:
            initial = lambda: np.concatenate((self._alpha.flatten(), self.rho.flatten(),
                                              self.base_coefficients.flatten(), self.components.flatten()))
        best_result = None
        best_fun = np.inf
        best_iteration = 0
        pbar = tqdm(range(multistarts))
        for i in pbar:
            result = minimize(fun=target_function, x0=initial(),
                              bounds=Bounds(self.lower_bound, self.upper_bound, keep_feasible=True),
                              options=kwargs, *args)
            if result.fun < best_fun:
                best_fun = result.fun
                best_result = result
                best_iteration = i
            pbar.set_description(
                f"Min loss: {np.round(best_fun, 3)} (iter={best_iteration}). Current loss: {np.round(result.fun, 3)}")

        if best_result is not None:
            target_function(best_result.x)
        else:
            target_function(initial())

        self.predictions = pd.DataFrame(self.predictions.T,
                                        columns=self.targets.columns,
                                        index=self.targets.index)
        if verbose:
            print(best_result)
        return self


class Decomposer(BaseDecomposer):
    def __init__(self, data: pd.DataFrame, branches: list, n_components: int = 2):
        super().__init__(data, branches, n_components)

    @property
    def common_base_coeff(self):
        return self.base_coefficients[0]

    @property
    def first_component_coeff(self):
        return self.base_coefficients[1]

    @property
    def second_component_coeff(self):
        assert self.n_components >= 2, "Второй компоненты не существует"
        return self.base_coefficients[2]

    @property
    def third_component_coeff(self):
        assert self.n_components >= 3, "Третьей компоненты не существует"
        return self.base_coefficients[3]

    @property
    def first_component(self):
        return self.components[0]

    @property
    def second_component(self):
        assert self.n_components >= 2, "Второй компоненты не существует"
        return self.components[1]

    @property
    def third_component(self):
        """
        :return:
        """
        assert self.n_components >= 3, "Третьей компоненты не существует"
        return self.components[2]

    def summary(self, mode='ces'):
        """
        Вывести статистики
        :param mode: ces -> rho, alpha, errors -> метрики
        :return:
        """
        mode = mode.lower()
        assert mode in ['ces', 'errors']
        if mode == 'ces':
            return pd.DataFrame(np.hstack((self.alpha, self.rho)).T,
                                index=[f'alpha_{i}' for i in range(self.n_components)] + ['rho'],
                                columns=self.branches)
        elif mode == 'errors':
            mean_squared_error = np.mean((self.targets - self.predictions) ** 2)
            mean_absolute_percentage_error = np.mean(
                np.abs((self.targets - self.predictions) / self.targets))
            mean_absolute_error = np.mean(np.abs(self.targets - self.predictions))
            return pd.DataFrame([mean_squared_error, mean_absolute_percentage_error, mean_absolute_error],
                                index=['MSE', "MAPE", "MAE"])

    def predict(self, branch=None):
        if branch is None:
            return self.predictions
        assert branch in self.targets.columns, "Отрасли нет среди таргета"
        return self.predictions[branch]

    def save(self, file_name: str = 'decomposer_res.npz'):
        """Сохранить результаты в файл"""
        np.savez(file_name, alpha=self.alpha, rho=self.rho, components=self.components,
                 base_coefficients=self.base_coefficients,
                 targets=self.targets.to_numpy())

    def __str__(self):
        return f"Decomposer with (n_components={self.n_components}, n_branches={self.n_branches})"

    def estimate_x(self, x, base_x=None):
        """
        :param base_x: изначальные значения
        :param x: матрица с объёмами продукции
        :return:
        """
        if base_x is None:
            base_x = x[0][:, None] * self.alpha
        p_X = self.targets.to_numpy().T
        coef0 = base_x / base_x[:, -1, None]
        coef1 = (self.alpha[:, -1][:, None] / self.alpha)
        coef2 = (self.components / self.components[-1])[None, :, :] * coef0[:, :, None]
        coef3 = 1 / (self.rho - 1)
        omega = ((coef2[:, :, :] * coef1[:, :, None]) ** coef3[:, None])
        omega = omega.transpose(1, 0, 2)
        nom = omega * base_x.T[:, :, None] * x.T * p_X[None, :, :]
        denom = np.sum(omega * self.components[:, None, :] * omega * base_x.T[:, :, None], axis=0)
        return nom / denom
