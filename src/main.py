import click
from runs import RunFactory
import os

@click.command(context_settings=dict(
    ignore_unknown_options=True,
))
@click.option('--run', '-r', default='adult_glofair', help='Run to execute')
@click.option('--project_name', '-p', default='AdultGlofair', help='Project name')
@click.option('--num_clients', '-n', default=10, help='Number of clients')
@click.option('--metric_name', '-m', default='demographic_parity', help='Metric name')
@click.option('--id', '-i', default='adult_glofair', help='Run id')
@click.option('--onlyperf', '-o', is_flag=True, help='Monitor only performance')
@click.option('--use_wandb', '-w', is_flag=True, help='Use wandb logger')
@click.option('--threshold', '-t', default=0.2, help='Fairness threshold')
@click.option('-metrics_list', '-ml', multiple=True, help='List of metrics')
@click.option('-groups_list', '-gl', multiple=True, help='List of groups')
@click.option('-threshold_list', '-tl', type=float, multiple=True, help='List of threshold')
@click.option('--experiment', '-e', default='alpha_09', help='Experiment name')
@click.option('--gpu_devices', '-g', multiple=True, help='List of GPU devices')
def main(run, project_name, num_clients, metric_name, id,
         use_wandb, onlyperf, threshold,gpu_devices,
         metrics_list, groups_list, threshold_list,experiment):
    """
    Main entry point for the CLI application.
    This function sets up and executes a run based on the provided command-line options.
    Parameters:
        run (str): The name of the run to execute.
        project_name (str): The name of the project.
        num_clients (int): The number of clients.
        metric_name (str): The name of the metric to use.
        id (str): The run identifier.
        use_wandb (bool): Flag to indicate whether to use wandb logger.
        onlyperf (bool): Flag to indicate whether to monitor only performance.
        threshold (float): The fairness threshold.
        gpu_devices (tuple): A tuple of GPU device identifiers.
        metrics_list (tuple): A tuple of metrics.
        groups_list (tuple): A tuple of groups.
        threshold_list (tuple): A tuple of thresholds.
        experiment (str): The name of the experiment.
    Returns:
        None
    """
    if gpu_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_devices)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
   
    run = RunFactory.create_run(run,
                                gpu_devices=gpu_devices,
                                project_name=project_name,
                                num_clients=num_clients,
                                experiment=experiment,
                                metric_name=metric_name,
                                id=id,
                                use_wandb=use_wandb,
                                onlyperf=onlyperf,
                                threshold=threshold,
                                metrics_list=metrics_list,
                                groups_list=groups_list,
                                threshold_list=threshold_list
                                )
    run()


if __name__ == '__main__':
    main()
