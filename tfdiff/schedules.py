# schedules.py
import numpy as np
from scipy.interpolate import interp1d

class DiffusionScheduler:
    """Manages different types of diffusion schedules"""
    
    @staticmethod
    def generate_schedules(config, num_steps):
        """Generate blur and noise schedules based on config"""
        # Check if schedule_config exists
        schedule_config = getattr(config, 'schedule_config', None)
        
        if not schedule_config:
            # Fall back to default linear schedules if no config provided
            return (
                DiffusionScheduler.linear_schedule(num_steps, 0.001, 0.1),  # Default blur schedule
                DiffusionScheduler.linear_schedule(num_steps, 1e-4, 0.05)   # Default noise schedule
            )
        
        # Handle both dictionary and AttrDict access
        blur_config = getattr(schedule_config, 'blur_schedule', {}) if hasattr(schedule_config, 'blur_schedule') else {}
        noise_config = getattr(schedule_config, 'noise_schedule', {}) if hasattr(schedule_config, 'noise_schedule') else {}
        
        # Generate each schedule independently
        blur_schedule = DiffusionScheduler.generate_single_schedule(
            schedule_type=blur_config.get('type', 'linear'),
            num_steps=num_steps,
            start=blur_config.get('start', 0.001),
            end=blur_config.get('end', 0.2),
            paper_values=blur_config.get('paper_values', None)
        )
        
        noise_schedule = DiffusionScheduler.generate_single_schedule(
            schedule_type=noise_config.get('type', 'linear'),
            num_steps=num_steps,
            start=noise_config.get('start', 0.0001),
            end=noise_config.get('end', 0.02),
            paper_values=noise_config.get('paper_values', None)
        )
        
        return blur_schedule, noise_schedule
    
    @staticmethod
    def generate_single_schedule(schedule_type, num_steps, start, end, paper_values=None):
        """Generate a single schedule based on type"""
        if schedule_type == "linear":
            return DiffusionScheduler.linear_schedule(num_steps, start, end)
            
        elif schedule_type == "cosine":
            return DiffusionScheduler.cosine_schedule(num_steps, start, end)
            
        elif schedule_type == "exponential":
            return DiffusionScheduler.exponential_schedule(num_steps, start, end)
            
        elif schedule_type == "paper":
            if paper_values is None:
                raise ValueError("Paper values must be provided for paper schedule type")
            return DiffusionScheduler.interpolate_paper_values(paper_values, num_steps)
            
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")

    @staticmethod
    def linear_schedule(num_steps, start, end):
        """Generate linear schedule"""
        return np.linspace(start, end, num_steps).tolist()
    
    @staticmethod
    def cosine_schedule(num_steps, start, end):
        """Generate cosine-based schedule"""
        t = np.linspace(0, 1, num_steps)
        alpha = 0.5 * (1 + np.cos(np.pi * t))
        return (start + (end - start) * (1 - alpha)).tolist()
    
    @staticmethod
    def exponential_schedule(num_steps, start, end):
        """Generate exponential schedule"""
        t = np.linspace(0, 1, num_steps)
        return (start * (end/start)**(t)).tolist()
    
    @staticmethod
    def interpolate_paper_values(paper_values, num_steps):
        """Interpolate paper values to desired number of steps"""
        x = np.linspace(0, 1, len(paper_values))
        y = np.array(paper_values)
        f = interp1d(x, y, kind='cubic')
        x_new = np.linspace(0, 1, num_steps)
        return f(x_new).tolist()