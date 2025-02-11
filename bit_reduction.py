import random
import torch
from utils import change_model_weights
import copy
class fitness: # progressive bit reduction algorithm (greedy)
    def __init__(self, model, data_loader, device, control_mode, ImageRecoder, trigger, mmd_loss, cosine_loss, mse_loss, parent_loader, fixed_fm, bitflip_info, greedy_scale, clean_trade_off):
        self.model = model
        self.origin_state_dict = copy.deepcopy(model.state_dict())
        self.data_loader = data_loader
        self.device = device
        self.control_mode = control_mode
        self.ImageRecoder = ImageRecoder
        self.trigger = trigger
        self.mmd_loss = mmd_loss
        self.cosine_loss = cosine_loss
        self.mse_loss = mse_loss
        self.parent_loader = parent_loader
        self.fixed_fm = fixed_fm
        self.record_dict = {}
        self.bitflip_info = bitflip_info
        self.greedy_scale = greedy_scale
        self.clean_trade_off = clean_trade_off
        print(f"greedy scale: {greedy_scale}")
    def run(self, individual, verbose=False):
        indexes = tuple([i for i, x in enumerate(self.bitflip_info) if x in individual])
        if indexes in self.record_dict.keys():
            print(f"find in record_dict, length: {len(individual)}")
            if verbose: print(self.record_dict[indexes][1])
            return self.record_dict[indexes][0]

        total_loss = 0.0
        clean_loss_total = 0.0
        mmd_loss_total = 0.0
        change_model_weights(self.model, individual, verbose=False)
        with torch.no_grad():
            for i, data in enumerate(self.data_loader):
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                poison_batch_inputs = self.ImageRecoder.sythesize_poison_image(inputs, self.trigger)

                if 'mmd' in self.control_mode:
                    if 'multi' in self.control_mode:
                        _, fm_list = self.model(poison_batch_inputs, latent=True, multi_latent=True)
                        _, fm_clean_list = self.model(inputs, latent=True, multi_latent=True)
                        loss_list = [1.0 - self.mmd_loss(fm_clean, fm) for fm_clean, fm in
                                     zip(fm_clean_list, fm_list)]
                        loss_mmd = sum(loss_list) / len(loss_list)
                        loss_3 = self.mmd_loss(fm_clean_list[-1], self.parent_loader.origin_fm[i], True)
                        loss = loss_mmd + 1.0 * loss_3

                    # elif 'single' in self.control_mode:
                    #     _, fm = self.model(poison_batch_inputs, latent=True)
                    #     _, fm_clean = self.model(inputs, latent=True)
                    #     loss_mmd = 1.0 - self.mmd_loss(fm_clean, fm)
                    #     loss_3 = self.mmd_loss(fm_clean, self.parent_loader.origin_fm[i], True)
                    #     loss = loss_mmd + 1.0 * loss_3

                    elif 'single' in self.control_mode:
                        _, fm = self.model(poison_batch_inputs, latent=True)
                        _, fm_clean = self.model(inputs, latent=True)
                        loss_mmd = 1.0 - self.mmd_loss(fm_clean, fm)
                        loss_3 = self.mmd_loss(fm_clean, self.parent_loader.origin_fm[0][i], True)
                        loss = loss_mmd + self.clean_trade_off * loss_3





                elif self.control_mode == 'fixed_fm':
                    _, fm = self.model(poison_batch_inputs, latent=True)
                    _, fm_clean = self.model(inputs, latent=True)
                    loss_mmd = torch.nn.MSELoss()(fm, self.fixed_fm)
                    loss_3 = torch.nn.MSELoss()(fm_clean, self.parent_loader.origin_fm[i])
                    loss = loss_mmd + 1.0 * loss_3
                elif self.control_mode == 'cosine':
                    _, fm = self.model(poison_batch_inputs, latent=True)
                    _, fm_clean = self.model(inputs, latent=True)
                    loss_mmd = 1.0 - self.cosine_loss(fm_clean, fm)
                    loss_3 = self.cosine_loss(fm_clean, self.parent_loader.origin_fm[i])
                    loss = loss_mmd + 1.0 * loss_3
                elif self.control_mode == 'mse':
                    _, fm = self.model(poison_batch_inputs, latent=True)
                    _, fm_clean = self.model(inputs, latent=True)
                    loss_mmd = 1.0 - self.mse_loss(fm_clean, fm)
                    loss_3 = self.mse_loss(fm_clean, self.parent_loader.origin_fm[i])
                    loss = loss_mmd + 1.0 * loss_3

                clean_loss_total += loss_3.item()
                mmd_loss_total += loss_mmd.item()
                total_loss += loss.detach().item()

            max_iter = len(self.data_loader)
            current_loss = (total_loss / max_iter)
            cur_clean_loss = (clean_loss_total / max_iter)
            cur_mmd_loss = (mmd_loss_total / max_iter)
        self.model.load_state_dict(self.origin_state_dict)
        score = 1.0 - current_loss + self.greedy_scale * (1.0 - current_loss) / len(individual)
        output = f"current loss: {current_loss:.3f}, " \
                 f"clean loss: {cur_clean_loss:.3f}, " \
                 f"mmd loss: {cur_mmd_loss:.3f}," \
                 f" length: {len(individual)}, " \
                 f"score: {score:.3f}"
        if verbose: print(output)


        self.record_dict[indexes] = (score, output)
        return score

class GeneticAlgorithm:
    def __init__(self, bitflip_info, population_size=100, mutation_rate=0.1, num_generations=100,
                 num_parents=50):
        self.bitflip_info = bitflip_info
        self.population_size = population_size
        self.max_chromosome_length = len(bitflip_info)
        self.min_chromosome_length = 1 #int(len(bitflip_info) / 5)
        self.mutation_rate = mutation_rate
        self.num_generations = num_generations
        self.num_parents = num_parents
        self.fitness = None

    def create_individual(self):
        length = random.randint(self.min_chromosome_length, self.max_chromosome_length)
        return random.sample(self.bitflip_info, length)

    def create_population(self):
        return [self.create_individual() for _ in range(self.population_size)]

    # def fitness(self, individual):
    #     total_loss = 0.0
    #     clean_loss_total = 0.0
    #     mmd_loss_total = 0.0
    #     data_loader = None
    #     self.model = None
    #     control_mode = None
    #     ImageRecoder = None
    #     trigger = None
    #
    #     with torch.no_grad():
    #         for i, data in enumerate(data_loader):
    #             inputs, labels = data[0].to(self.device), data[1].to(self.device)
    #             poison_batch_inputs = ImageRecoder.sythesize_poison_image(inputs, trigger)
    #
    #             if 'mmd' in control_mode:
    #                 if 'multi' in control_mode:
    #                     _, fm_list = self.model(poison_batch_inputs, latent=True, multi_latent=True)
    #                     _, fm_clean_list = self.model(inputs, latent=True, multi_latent=True)
    #                     loss_list = [1.0 - self.mmd_loss(fm_clean, fm) for fm_clean, fm in
    #                                  zip(fm_clean_list, fm_list)]
    #                     loss_mmd = sum(loss_list) / len(loss_list)
    #                     loss_3 = self.mmd_loss(fm_clean_list[-1], self.parent_loader.origin_fm[i], True)
    #                     loss = loss_mmd + 1.0 * loss_3
    #
    #                 elif 'single' in self.control_mode:
    #                     _, fm = self.model(poison_batch_inputs, latent=True)
    #                     _, fm_clean = self.model(inputs, latent=True)
    #                     loss_mmd = 1.0 - self.mmd_loss(fm_clean, fm)
    #                     loss_3 = self.mmd_loss(fm_clean, self.parent_loader.origin_fm[i], True)
    #                     loss = loss_mmd + 1.0 * loss_3
    #
    #             elif self.control_mode == 'fixed_fm':
    #                 _, fm = self.model(poison_batch_inputs, latent=True)
    #                 _, fm_clean = self.model(inputs, latent=True)
    #                 loss_mmd = torch.nn.MSELoss()(fm, self.fixed_fm)
    #                 loss_3 = torch.nn.MSELoss()(fm_clean, self.parent_loader.origin_fm[i])
    #                 loss = loss_mmd + 1.0 * loss_3
    #
    #
    #             elif self.control_mode == 'cosine':
    #                 _, fm = self.model(poison_batch_inputs, latent=True)
    #                 _, fm_clean = self.model(inputs, latent=True)
    #                 loss_mmd = 1.0 - self.cosine_loss(fm_clean, fm)
    #                 loss_3 = self.cosine_loss(fm_clean, self.parent_loader.origin_fm[i])
    #                 loss = loss_mmd + 1.0 * loss_3
    #
    #             clean_loss_total += loss_3.item()
    #             mmd_loss_total += loss_mmd.item()
    #
    #             total_loss += loss.detach().item()
    #
    #         max_iter = len(data_loader)
    #         current_loss = (total_loss / max_iter)
    #         cur_clean_loss = (clean_loss_total / max_iter)
    #         cur_mmd_loss = (mmd_loss_total / max_iter)

    def crossover(self, parent1, parent2):
        crossover_point1 = random.randint(1, len(parent1) - 1)
        crossover_point2 = random.randint(1, len(parent2) - 1)
        individual = parent1[:crossover_point1] + parent2[crossover_point2:]
        individual = [ x for x in self.bitflip_info if x in individual]
        return individual

    def mutate(self, individual):
        mutation_type = random.choice(["bit_flip", "add_gene", "remove_gene"])

        if mutation_type == "add_gene" and len(individual) < self.max_chromosome_length:
            excluded_genes = [gene for gene in self.bitflip_info if gene not in individual]
            random_element = random.choice(excluded_genes)
            index = individual.index(individual[-1])
            individual.insert(index, random_element)
            individual = [ x for x in self.bitflip_info if x in individual]
        elif mutation_type == "remove_gene" and len(individual) > self.min_chromosome_length:
            index = random.randint(0, len(individual) - 1)
            individual.pop(index)

        return individual

    def selection(self, population):

        sorted_population = sorted(population, key=self.fitness.run, reverse=True)
        print(f"Best case for current generation ==>")
        self.fitness.run(sorted_population[0], verbose=True)
        return sorted_population[:self.num_parents]

    def run(self):
        population = self.create_population()

        for generation in range(self.num_generations):
            print(f"########################Generation: {generation:>3}########################")
            selected_parents = self.selection(population)
            offspring = []

            for _ in range(self.population_size):
                parent1, parent2 = random.sample(selected_parents, 2)
                child = self.crossover(parent1, parent2)
                mutated_child = self.mutate(child)
                offspring.append(mutated_child)

            population = offspring

        best_individual = max(population, key=self.fitness.run)
        return best_individual, self.fitness.run(best_individual, verbose=True)

