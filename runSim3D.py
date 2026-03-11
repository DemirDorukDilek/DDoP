"""

DDoP Live Drone Simulator
=========================

Calistir:  python sim_app.py

Tarayicida: http://localhost:8050

"""


import numpy as np

from dash import Dash, html, dcc, Patch

from dash.dependencies import Input, Output, State

import plotly.graph_objects as go

from scipy.spatial import ConvexHull


from DDOP.CorridorGeneration.IRIS import greedy_corridor_generation

from DDOP.CorridorGeneration.RRT import rrt

from DDOP.DDoP.Optimizer import optimize_with_split

from DDOP.Visualization3d import _get_trajectory_points

from DDOP.Utils import H2V

from DDOP.CorridorGeneration.Corridor_Utils import HPolyhedron


# ===================== SAHNE AYARLARI =====================


BOUNDS = (np.array([0., 0., 0.]), np.array([10., 10., 10.]))


OBSTACLES = [

    HPolyhedron.from_Vrep([[3,0,0],[4,0,0],[4,4,0],[3,4,0],[3,0,4],[4,0,4],[4,4,4],[3,4,4]]),

    HPolyhedron.from_Vrep([[3,6,0],[4,6,0],[4,10,0],[3,10,0],[3,6,4],[4,6,4],[4,10,4],[3,10,4]]),

    HPolyhedron.from_Vrep([[6,2,0],[8,2,0],[8,5,0],[6,5,0],[6,2,5],[8,2,5],[8,5,5],[6,5,5]]),

    HPolyhedron.from_Vrep([[7,7,3],[8.5,7,3],[8.5,8.5,3],[7,8.5,3],[7,7,7],[8.5,7,7],[8.5,8.5,7],[7,8.5,7]]),
]


START_POS = np.array([0.5, 9.5, 2.0])


# ===================== ONCEDEN HESAPLANAN ENGEL VERILERI =====================


OBSTACLE_DATA = []

for _obs in OBSTACLES:

    _verts = H2V(_obs.A, _obs.b)

    if _verts is not None and len(_verts) >= 4:

        try:

            _hull = ConvexHull(_verts)

            OBSTACLE_DATA.append({

                'x': _verts[:, 0].tolist(),

                'y': _verts[:, 1].tolist(),

                'z': _verts[:, 2].tolist(),

                'i': _hull.simplices[:, 0].tolist(),

                'j': _hull.simplices[:, 1].tolist(),

                'k': _hull.simplices[:, 2].tolist(),

            })

        except Exception:
            pass


N_OBSTACLES = len(OBSTACLE_DATA)


# ===================== YARDIMCI FONKSIYONLAR =====================


def plan_and_get_trajectory(start, goal):

    """RRT + IRIS + Optimization — hizli parametreler."""

    path = rrt(start, goal, OBSTACLES, BOUNDS, max_iter=3000, step_size=1.0)

    if path is None:

        return None, "RRT basarisiz!"


    iris_result = greedy_corridor_generation(path, OBSTACLES, BOUNDS)

    if iris_result is None:

        return None, "Koridor olusturulamadi!"


    corridors, waypoints, radii = iris_result

    hpolys = [(c.hpoly.A, c.hpoly.b) for c in corridors]


    opt, Ts, op_wp, opt_hpolys = optimize_with_split(

        [1.0] * (len(waypoints) - 1), waypoints, hpolys, 5
    )


    t_all, coords = _get_trajectory_points(opt, num_points=200)


    poly_data = []

    n_poly = len(opt_hpolys)

    colors = [f'hsla({int(i*360/max(n_poly,1))}, 70%, 60%, 0.2)' for i in range(n_poly)]

    for idx, (A, b) in enumerate(opt_hpolys):

        verts = H2V(A, b)

        if verts is not None and len(verts) >= 4:

            try:

                hull = ConvexHull(verts)

                poly_data.append({

                    'x': verts[:, 0].tolist(), 'y': verts[:, 1].tolist(), 'z': verts[:, 2].tolist(),

                    'i': hull.simplices[:, 0].tolist(), 'j': hull.simplices[:, 1].tolist(),

                    'k': hull.simplices[:, 2].tolist(),

                    'color': colors[idx], 'name': f'P{idx}',

                })

            except Exception:
                pass


    return {
        't': t_all.tolist(),

        'x': coords[0].tolist(), 'y': coords[1].tolist(), 'z': coords[2].tolist(),

        'poly_data': poly_data,

    }, "OK"



def build_full_figure(drone_pos, trail_x, trail_y, trail_z, poly_data=None, status="Hazir"):

    """

    Tum figure'u sifirdan olustur.

    Trace sirasi: [obstacles...] [polytopes...] [trail] [drone]

    Trail index = N_OBSTACLES + len(poly_data)

    Drone index = trail_index + 1

    """
    traces = []


    # Engeller

    for oi, od in enumerate(OBSTACLE_DATA):

        traces.append(go.Mesh3d(

            x=od['x'], y=od['y'], z=od['z'],

            i=od['i'], j=od['j'], k=od['k'],

            color='rgba(180, 60, 60, 0.6)', flatshading=True,

            name=f'Obstacle {oi}', showlegend=(oi == 0),

            legendgroup='obstacles', hoverinfo='name',
        ))


    # Polytope'lar

    if poly_data:

        for p in poly_data:

            traces.append(go.Mesh3d(

                x=p['x'], y=p['y'], z=p['z'],

                i=p['i'], j=p['j'], k=p['k'],

                color=p['color'], flatshading=True,

                name=p['name'], showlegend=True, hoverinfo='name',
            ))


    # Trail (bos olsa bile trace olarak ekle — Patch ile guncelleyecegiz)

    traces.append(go.Scatter3d(

        x=trail_x, y=trail_y, z=trail_z,

        mode='lines', line=dict(color='red', width=4),

        name='Trail', showlegend=True, hoverinfo='skip',
    ))


    # Drone

    traces.append(go.Scatter3d(

        x=[drone_pos[0]], y=[drone_pos[1]], z=[drone_pos[2]],

        mode='markers+text',

        marker=dict(size=8, color='lime', symbol='diamond',

                    line=dict(width=2, color='black')),

        text=['Drone'], textposition='top center',

        name='Drone', showlegend=True,

        hovertext=f'X={drone_pos[0]:.2f} Y={drone_pos[1]:.2f} Z={drone_pos[2]:.2f}',

        hoverinfo='text',
    ))


    fig = go.Figure(data=traces)

    lower, upper = BOUNDS

    m = 0.5

    fig.update_layout(
        scene=dict(
            aspectmode='data',

            xaxis_title='X', yaxis_title='Y', zaxis_title='Z',

            xaxis=dict(range=[lower[0]-m, upper[0]+m]),

            yaxis=dict(range=[lower[1]-m, upper[1]+m]),

            zaxis=dict(range=[lower[2]-m, upper[2]+m]),
        ),

        height=700, width=900,

        margin=dict(l=0, r=0, t=40, b=0),

        title=f'DDoP Live Sim — {status}',

        legend=dict(x=1.02, y=1, font=dict(size=10)),

        uirevision='keep',
    )

    return fig



# ===================== DASH APP =====================


app = Dash(__name__)


app.layout = html.Div([

    html.Div([

        html.H2("DDoP Drone Simulator", style={'margin': '0', 'color': '#2c3e50'}),

        html.Div([

            html.Div([

                html.Label("X:", style={'fontWeight': 'bold'}),

                dcc.Input(id='target-x', type='number', value=9.0, step=0.5,

                          style={'width': '70px', 'marginRight': '10px'}),

            ], style={'display': 'inline-block'}),

            html.Div([

                html.Label("Y:", style={'fontWeight': 'bold'}),

                dcc.Input(id='target-y', type='number', value=4.0, step=0.5,

                          style={'width': '70px', 'marginRight': '10px'}),

            ], style={'display': 'inline-block'}),

            html.Div([

                html.Label("Z:", style={'fontWeight': 'bold'}),

                dcc.Input(id='target-z', type='number', value=4.0, step=0.5,

                          style={'width': '70px', 'marginRight': '10px'}),

            ], style={'display': 'inline-block'}),

            html.Button("Git!", id='go-btn',

                        style={'padding': '8px 24px', 'fontSize': '16px',

                               'backgroundColor': '#27ae60', 'color': 'white',

                               'border': 'none', 'borderRadius': '5px',

                               'cursor': 'pointer', 'marginLeft': '15px'}),

            html.Button("Reset", id='reset-btn',

                        style={'padding': '8px 16px', 'fontSize': '14px',

                               'backgroundColor': '#e74c3c', 'color': 'white',

                               'border': 'none', 'borderRadius': '5px',

                               'cursor': 'pointer', 'marginLeft': '10px'}),

            html.Div([

                html.Label("Hiz:", style={'fontWeight': 'bold', 'marginLeft': '15px'}),

                dcc.Slider(id='speed-slider', min=1, max=10, step=1, value=4,

                           marks={1: '1x', 4: '4x', 7: '7x', 10: '10x'},

                           tooltip={"placement": "bottom"}),

            ], style={'display': 'inline-block', 'width': '200px', 'verticalAlign': 'middle'}),

        ], style={'marginTop': '10px'}),

        html.Div(id='status-text',

                 style={'marginTop': '8px', 'fontSize': '14px', 'color': '#7f8c8d'}),

    ], style={'padding': '15px', 'backgroundColor': '#ecf0f1', 'borderBottom': '2px solid #bdc3c7'}),


    dcc.Graph(id='sim-graph', style={'height': '75vh'}),

    dcc.Interval(id='anim-interval', interval=50, disabled=True),


    # Stores

    dcc.Store(id='drone-state', data={

        'pos': START_POS.tolist(),

        'trail_x': [START_POS[0]], 'trail_y': [START_POS[1]], 'trail_z': [START_POS[2]],

    }),

    dcc.Store(id='traj-data', data=None),

    dcc.Store(id='anim-idx', data=0),

    dcc.Store(id='trace-indices', data={'trail': N_OBSTACLES, 'drone': N_OBSTACLES + 1}),

], style={'fontFamily': 'Arial, sans-serif'})



# --- Git! butonu: planla + full figure olustur ---

@app.callback(

    Output('sim-graph', 'figure', allow_duplicate=True),

    Output('traj-data', 'data'),

    Output('anim-idx', 'data', allow_duplicate=True),

    Output('anim-interval', 'disabled', allow_duplicate=True),

    Output('status-text', 'children', allow_duplicate=True),

    Output('trace-indices', 'data', allow_duplicate=True),

    Input('go-btn', 'n_clicks'),

    State('target-x', 'value'), State('target-y', 'value'), State('target-z', 'value'),

    State('drone-state', 'data'),

    prevent_initial_call=True,
)

def on_go(n_clicks, tx, ty, tz, drone_state):

    if tx is None or ty is None or tz is None:

        from dash import no_update

        return no_update, None, 0, True, "Koordinatlari gir!", no_update


    start = np.array(drone_state['pos'])

    goal = np.array([float(tx), float(ty), float(tz)])


    if np.linalg.norm(goal - start) < 0.3:

        from dash import no_update

        return no_update, None, 0, True, "Zaten oradasin!", no_update


    lower, upper = BOUNDS

    for i, (ax, lo, hi) in enumerate(zip(['X','Y','Z'], lower, upper)):

        if goal[i] < lo or goal[i] > hi:

            from dash import no_update

            return no_update, None, 0, True, f"{ax}={goal[i]:.1f} sinir disinda!", no_update


    result, msg = plan_and_get_trajectory(start, goal)

    if result is None:

        from dash import no_update

        return no_update, None, 0, True, f"Hata: {msg}", no_update


    poly_data = result.pop('poly_data')

    n_poly = len(poly_data)


    # Trail & drone indexleri

    trail_idx = N_OBSTACLES + n_poly

    drone_idx = trail_idx + 1


    trail_x = drone_state['trail_x']

    trail_y = drone_state['trail_y']

    trail_z = drone_state['trail_z']


    fig = build_full_figure(

        drone_state['pos'], trail_x, trail_y, trail_z,

        poly_data, f"Ucuyor! -> ({tx:.1f}, {ty:.1f}, {tz:.1f})"
    )


    return (fig, result, 0, False,

            f"Ucuyor! -> ({tx:.1f}, {ty:.1f}, {tz:.1f})",

            {'trail': trail_idx, 'drone': drone_idx})



# --- Reset ---

@app.callback(

    Output('sim-graph', 'figure', allow_duplicate=True),

    Output('drone-state', 'data', allow_duplicate=True),

    Output('traj-data', 'data', allow_duplicate=True),

    Output('anim-idx', 'data', allow_duplicate=True),

    Output('anim-interval', 'disabled', allow_duplicate=True),

    Output('status-text', 'children', allow_duplicate=True),

    Output('trace-indices', 'data', allow_duplicate=True),

    Input('reset-btn', 'n_clicks'),

    prevent_initial_call=True,
)

def on_reset(n):

    pos = START_POS.tolist()

    state = {'pos': pos, 'trail_x': [pos[0]], 'trail_y': [pos[1]], 'trail_z': [pos[2]]}

    fig = build_full_figure(pos, [pos[0]], [pos[1]], [pos[2]], None, "Reset!")

    indices = {'trail': N_OBSTACLES, 'drone': N_OBSTACLES + 1}

    return fig, state, None, 0, True, "Reset! Drone baslangica dondu.", indices



# --- Animasyon: Patch ile sadece trail + drone guncelle ---

@app.callback(

    Output('sim-graph', 'figure'),

    Output('drone-state', 'data'),

    Output('anim-idx', 'data'),

    Output('anim-interval', 'disabled'),

    Output('status-text', 'children'),

    Input('anim-interval', 'n_intervals'),

    State('drone-state', 'data'),

    State('traj-data', 'data'),

    State('anim-idx', 'data'),

    State('trace-indices', 'data'),

    State('speed-slider', 'value'),
)

def animate(n_intervals, drone_state, traj_data, anim_idx, indices, speed):
    pos = drone_state['pos']

    trail_x = drone_state['trail_x']

    trail_y = drone_state['trail_y']

    trail_z = drone_state['trail_z']


    if traj_data is None or anim_idx >= len(traj_data['x']) - 1:

        from dash import no_update

        return no_update, drone_state, anim_idx, True, no_update


    # Ilerleme (speed slider ile kontrol)

    step = min(anim_idx + speed, len(traj_data['x']) - 1)

    cx = traj_data['x'][step]

    cy = traj_data['y'][step]

    cz = traj_data['z'][step]


    trail_x.append(cx)

    trail_y.append(cy)

    trail_z.append(cz)


    # Patch: sadece trail + drone guncelle (TUM FIGURU REBUILD ETME)

    patch = Patch()
    ti = indices['trail']
    di = indices['drone']


    patch['data'][ti]['x'] = trail_x

    patch['data'][ti]['y'] = trail_y

    patch['data'][ti]['z'] = trail_z

    patch['data'][di]['x'] = [cx]

    patch['data'][di]['y'] = [cy]

    patch['data'][di]['z'] = [cz]

    patch['data'][di]['hovertext'] = f'X={cx:.2f} Y={cy:.2f} Z={cz:.2f}'


    t_val = traj_data['t'][step]

    done = step >= len(traj_data['x']) - 1

    status = (f"Vardi! ({cx:.1f}, {cy:.1f}, {cz:.1f})" if done

              else f"t={t_val:.2f}s | X={cx:.2f} Y={cy:.2f} Z={cz:.2f}")


    new_state = {'pos': [cx, cy, cz], 'trail_x': trail_x, 'trail_y': trail_y, 'trail_z': trail_z}

    return patch, new_state, step, done, status



if __name__ == '__main__':

    print("\n" + "="*50)

    print("  DDoP Drone Simulator")

    print("  http://localhost:8050")

    print("="*50 + "\n")

    app.run(debug=False, port=8050)

