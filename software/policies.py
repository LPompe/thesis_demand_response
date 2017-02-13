from simulation_enviroment import Cell


class BasePolicy():

    def __init__(self):
        pass

    def policy(self, s, cells):
        return [Cell.off for _ in cells]



class LatestSwitchPolicy(BasePolicy):

    def policy(self, s, cells):
        cell_policies = []
        for cell in cells:
            if cell.temperature >= cell.max_temp:
                cell_policies.append(Cell.on)

            elif cell.temperature <= cell.min_temp:
                cell_policies.append(Cell.off)
                
            else:
                cell_policies.append(cell.state)

        return cell_policies
