from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.descriptions import ComposableNode
from launch_ros.actions import ComposableNodeContainer
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Get the share directory of my_py_amcl
    my_py_amcl_share_dir = get_package_share_directory('my_py_amcl')
    
    # Get the share directory of turtlebot3_gazebo
    turtlebot3_gazebo_share_dir = get_package_share_directory('turtlebot3_gazebo')
    
    # Path to the maze launch file
    maze_launch_file = os.path.join(
        turtlebot3_gazebo_share_dir, 'launch', 'turtlebot3_maze.launch.py'
    )

    # Path to the map file from our package
    map_file = os.path.join(
        my_py_amcl_share_dir, 'maps', 'map.yaml'
    )

    # Path to our rviz config
    rviz_config_file = os.path.join(
        my_py_amcl_share_dir, 'rviz', 'amcl.rviz'
    )

    costmap_params_file = os.path.join(
        my_py_amcl_share_dir, 'config', 'costmap.yaml'
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='True',
            description='Use simulation (Gazebo) clock if true'),

        # Include the Gazebo simulation launch file
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(maze_launch_file)
        ),

        ComposableNodeContainer(
            name='nav2_container',
            namespace='',
            package='rclcpp_components',
            executable='component_container',
            composable_node_descriptions=[
                ComposableNode(
                    package='nav2_map_server',
                    plugin='nav2_map_server::MapServer',
                    name='map_server',
                    parameters=[
                        {'use_sim_time': LaunchConfiguration('use_sim_time')},
                        {'yaml_filename': map_file}
                    ]
                ),
            ],
            output='screen',
        ),

        # Launch our Python AMCL node
        Node(
            package='my_py_amcl',
            executable='amcl_node',
            name='my_py_amcl',
            output='screen',
            parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}]
        ),

        TimerAction(
            period=3.0,
            actions=[
                Node(
                    package='nav2_lifecycle_manager',
                    executable='lifecycle_manager',
                    name='lifecycle_manager_localization',
                    output='screen',
                    parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')},
                                {'autostart': True},
                                {'node_names': ['map_server']}]
                ),
            ]
        ),

        # Launch RViz2
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', rviz_config_file],
            parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}]
        ),
    ]) 