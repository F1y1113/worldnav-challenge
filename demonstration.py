#!/usr/bin/env python3
"""Illustrative WorldNav five-metric calculation on invented data only.

This standalone demonstration includes no hidden challenge reference, no
submission parser and no CodaBench phase runner. The synthetic example at the
end shows the expected input shape; real submissions contain actions only.
"""

import bisect
import math


EPS = 1e-12
GAUSS_X = (-0.9894009349916499,-0.9445750230732326,-0.8656312023878318,-0.755404408355003,-0.6178762444026438,-0.45801677765722737,-0.2816035507792589,-0.09501250983763744,0.09501250983763744,0.2816035507792589,0.45801677765722737,0.6178762444026438,0.755404408355003,0.8656312023878318,0.9445750230732326,0.9894009349916499)
GAUSS_W = (0.027152459411753902,0.062253523938647824,0.09515851168249272,0.12462897125553399,0.1495959888165767,0.16915651939500265,0.1826034150449236,0.1894506104550685,0.1894506104550685,0.1826034150449236,0.16915651939500265,0.1495959888165767,0.12462897125553399,0.09515851168249272,0.062253523938647824,0.027152459411753902)



def _edge_free(point, a, b, radius):
    vx, vy = b[0]-a[0], b[1]-a[1]
    wx, wy = a[0]-point[0], a[1]-point[1]
    aa = vx*vx + vy*vy
    cc = wx*wx + wy*wy - radius*radius
    if aa < 1e-24:
        return (0., 1.) if cc <= 1e-13 else None
    bb = 2*(wx*vx + wy*vy)
    disc = bb*bb - 4*aa*cc
    if disc < -1e-13:
        return None
    disc = max(0., disc)
    root = math.sqrt(disc)
    lo = max(0., (-bb-root)/(2*aa))
    hi = min(1., (-bb+root)/(2*aa))
    return (lo, hi) if lo <= hi + 1e-12 else None


def _trim(interval, lower):
    if interval is None:
        return None
    lo = max(interval[0], lower)
    return (lo, interval[1]) if lo <= interval[1] + 1e-12 else None


def _outer_boundary(point, vertices, radius):
    out = []
    end_reachable = True
    for a, b in zip(vertices, vertices[1:]):
        free = _edge_free(point, a, b, radius)
        reachable = (0., free[1]) if (end_reachable and free is not None
                                    and free[0] <= 1e-12) else None
        out.append(reachable)
        end_reachable = (reachable is not None and
                         reachable[1] >= 1. - 1e-12)
    return out


def _decide(pred, ref, radius):
    if math.dist(pred[0], ref[0]) > radius + 1e-12:
        return False
    if math.dist(pred[-1], ref[-1]) > radius + 1e-12:
        return False
    if len(pred) == 1:
        return all(math.dist(pred[0], r) <= radius + 1e-12 for r in ref)
    if len(ref) == 1:
        return all(math.dist(p, ref[0]) <= radius + 1e-12 for p in pred)

    left_boundary = _outer_boundary(pred[0], ref, radius)
    bottom_boundary = _outer_boundary(ref[0], pred, radius)
    previous_right = [None] * (len(ref)-1)
    last_top = last_right = None
    for i in range(len(pred)-1):
        current_right = [None] * (len(ref)-1)
        previous_top = None
        for j in range(len(ref)-1):
            left = (left_boundary[j] if i == 0 else previous_right[j])
            bottom = (bottom_boundary[i] if j == 0 else previous_top)
            right_free = _edge_free(pred[i+1], ref[j], ref[j+1], radius)
            top_free = _edge_free(ref[j+1], pred[i], pred[i+1], radius)
            right = (right_free if bottom is not None else
                     _trim(right_free, left[0]) if left is not None else None)
            top = (top_free if left is not None else
                   _trim(top_free, bottom[0]) if bottom is not None else None)
            current_right[j] = right
            previous_top = top
            if i == len(pred)-2 and j == len(ref)-2:
                last_top, last_right = top, right
        previous_right = current_right
    return (last_top is not None and last_top[1] >= 1.-1e-12 or
            last_right is not None and last_right[1] >= 1.-1e-12)


def _canonical(points):
    out = []
    for p in points:
        p = (float(p[0]), float(p[1]))
        if not out or math.dist(p, out[-1]) > 1e-12:
            out.append(p)
    return out


def bracketed_frechet(pred, ref, absolute_precision=1e-8):
    pred, ref = _canonical(pred), _canonical(ref)
    if not pred or not ref:
        raise ValueError('empty curve')
    if pred == ref:
        return 0., 0.
    lo = max(math.dist(pred[0], ref[0]), math.dist(pred[-1], ref[-1]))
    hi = max(math.dist(p, r) for p in pred for r in ref)
    if not _decide(pred, ref, hi):
        raise AssertionError('full-diameter upper bound must be feasible')
    while hi-lo > absolute_precision:
        middle = (lo+hi)/2.
        if _decide(pred, ref, middle):
            hi = middle
        else:
            lo = middle
    return lo, hi


def bracketed_frechet_scaled(pred, ref, relative_precision=1e-9):
    if not pred or not ref:
        raise ValueError('empty curve')
    origin = ref[0]
    shifted_pred = [(float(x)-float(origin[0]), float(y)-float(origin[1]))
                    for x, y in pred]
    shifted_ref = [(float(x)-float(origin[0]), float(y)-float(origin[1]))
                   for x, y in ref]
    scale = max(math.hypot(x, y) for x, y in shifted_pred + shifted_ref)
    if scale == 0.:
        return 0., 0.
    pred_unit = [(x/scale, y/scale) for x, y in shifted_pred]
    ref_unit = [(x/scale, y/scale) for x, y in shifted_ref]
    lo, hi = bracketed_frechet(pred_unit, ref_unit,
                               absolute_precision=relative_precision)
    return scale*lo, scale*hi


def _bbox_lower_bound(predicted_xy, reference_xy):
    xmin=min(x for x,_ in reference_xy)
    xmax=max(x for x,_ in reference_xy)
    ymin=min(y for _,y in reference_xy)
    ymax=max(y for _,y in reference_xy)
    return max(math.hypot(max(xmin-x,0.,x-xmax),
                          max(ymin-y,0.,y-ymax)) for x,y in predicted_xy)


def _move_pieces(xy, yaw):
    out=[]; length=0.
    for i in range(len(xy)-1):
        amount=math.dist(xy[i],xy[i+1])
        if amount > EPS:
            out.append((length,length+amount,float(yaw[i]),float(yaw[i+1])))
            length += amount
    return out,length


def _yaw_at(pieces, ends, last_yaw, s):
    if not pieces or s >= pieces[-1][1]:
        return last_yaw
    j=bisect.bisect_right(ends,s)
    if j>=len(pieces):
        return last_yaw
    a,b,y0,y1=pieces[j]
    return y0+(y1-y0)*(s-a)/(b-a)


def _move_error(px,py,rx,ry):
    pp,lp=_move_pieces(px,py)
    rr,lr=_move_pieces(rx,ry)
    if lr <= EPS:
        return (1.-math.cos(py[-1]-ry[-1]))/2., lp, lr
    cuts={0.,max(lp,lr)}
    for a,b,_,_ in pp+rr:
        cuts.add(a);cuts.add(b)
    cuts=sorted(cuts)
    pe=[p[1] for p in pp];re=[p[1] for p in rr]
    integral=0.
    for a,b in zip(cuts,cuts[1:]):
        if b-a<=EPS:
            continue
        gap_a=(_yaw_at(pp,pe,py[-1],a+EPS*(b-a))-
               _yaw_at(rr,re,ry[-1],a+EPS*(b-a)))
        gap_b=(_yaw_at(pp,pe,py[-1],b-EPS*(b-a))-
               _yaw_at(rr,re,ry[-1],b-EPS*(b-a)))
        rate=(gap_b-gap_a)/(b-a)
        if abs(rate)<1e-10:
            integral+=(b-a)*(1.-math.cos((gap_a+gap_b)/2.))/2.
        else:
            integral+=(b-a)/2.-(math.sin(gap_b)-math.sin(gap_a))/(2.*rate)
    return integral/lr,lp,lr


def _turn_pieces(xy,yaw):
    out=[];total=0.
    for i in range(len(yaw)-1):
        change=float(yaw[i+1]-yaw[i])
        size=abs(change)
        if size<=EPS:
            continue
        out.append((total,total+size,xy[i],xy[i+1],
                    float(yaw[i]),float(yaw[i+1]),
                    1 if change>0 else -1))
        total+=size
    return out,total


def _turn_state(piece,at):
    a,b,x0,x1,y0,y1,sgn=piece
    frac=(at-a)/(b-a)
    return ((x0[0]+(x1[0]-x0[0])*frac,
             x0[1]+(x1[1]-x0[1])*frac),
            y0+(y1-y0)*frac,sgn)


def _turn_cost(px,py,rx,ry,spatial_scale):
    pp,tp=_turn_pieces(px,py)
    rr,tr=_turn_pieces(rx,ry)
    if max(tp,tr)<=EPS:
        return 0.,tp,tr
    cuts={0.,max(tp,tr)}
    for a,b,*_ in pp+rr:
        cuts.add(a);cuts.add(b)
    cuts=sorted(cuts)
    pe=[p[1] for p in pp];re=[p[1] for p in rr]
    integral=0.
    for a,b in zip(cuts,cuts[1:]):
        if b-a<=EPS:
            continue
        mid=(a+b)/2.
        ip=bisect.bisect_right(pe,mid)
        ir=bisect.bisect_right(re,mid)
        if ip>=len(pp) or ir>=len(rr) or pp[ip][-1]!=rr[ir][-1]:
            integral+=b-a
            continue
        half=(b-a)/2.
        for node,weight in zip(GAUSS_X,GAUSS_W):
            x1,y1,_=_turn_state(pp[ip],mid+half*float(node))
            x2,y2,_=_turn_state(rr[ir],mid+half*float(node))
            distance_sq=(x1[0]-x2[0])**2+(x1[1]-x2[1])**2
            loss=1.-math.exp(-distance_sq/(spatial_scale*spatial_scale))*(1.+math.cos(y1-y2))/2.
            integral+=half*float(weight)*loss
    return integral/(2.*math.pi),tp,tr


def measure_h(pred_xy,pred_yaw,ref_xy,ref_yaw):
    emove,lp,lr=_move_error(pred_xy,pred_yaw,ref_xy,ref_yaw)
    x0,y0=ref_xy[0]
    span=max(math.hypot(x-x0,y-y0) for x,y in ref_xy)
    # On a purely stationary reference, turn-location separations vanish for
    # exact-position predictions. An off-position prediction is already
    # penalized by the route component; avoid division by zero here.
    spatial_scale=span if span>1e-12 else 1.
    cturn,tp,tr=_turn_cost(pred_xy,pred_yaw,ref_xy,ref_yaw,spatial_scale)
    h=max(0.,1.-emove)/(1.+cturn)
    if not (math.isfinite(h) and 0<=h<=1+1e-10):
        raise ValueError(f'out-of-range H: {h}')
    return {'H':min(1.,h),'E_move':emove,'C_turn':cturn,
            'pred_translation_units':lp,'ref_translation_units':lr,
            'pred_absolute_turn_rad':tp,'ref_absolute_turn_rad':tr}


def canonical_yaws(yaws):
    if not yaws:
        raise ValueError('empty yaw sequence')
    result=[float(yaws[0])]
    for a,b in zip(yaws,yaws[1:]):
        d=math.atan2(math.sin(float(b)-float(a)),
                     math.cos(float(b)-float(a)))
        if abs(d)<=1e-12:
            d=0.
        if abs(abs(d)-math.pi)<=1e-10:
            d=math.pi
        result.append(result[-1]+d)
    return result



def compose(initial_pose, actions):
    x, y, yaw = map(float, initial_pose[:3])
    positions, headings = [(x, y)], [yaw]
    for action in actions:
        dx, dy, dyaw = map(float, action[:3])
        x, y, yaw = x+dx, y+dy, yaw+dyaw
        positions.append((x, y))
        headings.append(yaw)
    return positions, headings


def route_facing(reference_xy, reference_yaw, pred_xy, pred_yaw):
    ref = [tuple(p[:2]) for p in reference_xy]
    pred = [tuple(p[:2]) for p in pred_xy]
    x0, y0 = ref[0]
    rho = max(math.hypot(x-x0, y-y0) for x, y in ref)
    lower = _bbox_lower_bound(pred, ref)
    if rho > 1e-12 and lower >= rho:
        d = 0.0
    else:
        _, upper = bracketed_frechet_scaled(pred, ref)
        d = max(0.0, 1.0-upper/rho) if rho > 1e-12 else math.exp(-upper/.025)
    lr = math.fsum(math.dist(a,b) for a,b in zip(ref,ref[1:]))
    lp = math.fsum(math.dist(a,b) for a,b in zip(pred,pred[1:]))
    d *= min(1.0, lr/lp) if lp > 1e-12 else 1.0
    h = measure_h(pred, pred_yaw, ref, canonical_yaws(reference_yaw))["H"]
    return d, h


def episode(reference, actions):
    ref_xy = reference["positions"]
    ref_yaw = reference["yaws"]
    pred_xy, pred_yaw = compose(reference["initial_pose"], actions)
    d, h = route_facing(ref_xy, ref_yaw, pred_xy, pred_yaw)
    t = len(ref_xy)-1
    if t < 1 or len(ref_yaw) != t+1:
        raise ValueError("malformed reference")
    last = len(pred_xy)-1
    ate2, rpe2 = [], []
    for j in range(1, t+1):
        current, previous = pred_xy[min(j,last)], pred_xy[min(j-1,last)]
        desired, desired_previous = ref_xy[j], ref_xy[j-1]
        ex, ey = current[0]-desired[0], current[1]-desired[1]
        ate2.append(ex*ex+ey*ey)
        ux = (current[0]-previous[0])-(desired[0]-desired_previous[0])
        uy = (current[1]-previous[1])-(desired[1]-desired_previous[1])
        yaw = ref_yaw[j-1]
        c, s = math.cos(yaw), math.sin(yaw)
        lx, ly = c*ux+s*uy, -s*ux+c*uy
        rpe2.append(lx*lx+ly*ly)
    ate_sum, rpe_sum = math.fsum(ate2), math.fsum(rpe2)
    a, r = math.sqrt(ate_sum/t), math.sqrt(rpe_sum/t)
    goal = reference["goal_pose"]
    success = int(math.dist(pred_xy[-1],goal[:2]) <
                  max(reference["avg_step_length"],.6))
    b = d*(1+h)/2
    penalty = .10*(1-success)+.10*a/(1+a)+.10*r/(1+r)
    points = 100*b*(1-penalty)
    if not all(math.isfinite(v) for v in (ate_sum,rpe_sum,d,h,points)):
        raise ValueError("non-finite score")
    if not (0<=d<=1 and 0<=h<=1 and 0<=points<=100.000001):
        raise ValueError("out-of-range score")
    return {"success":success,"ate_squared_sum":ate_sum,
            "rpe_squared_sum":rpe_sum,"reference_boundaries":t,
            "D":d,"H":h,"score_points":points}


def score_split(reference_episodes, predictions):
    if set(reference_episodes)!=set(predictions):
        raise ValueError("episode coverage mismatch")
    rows=[episode(reference_episodes[sid],predictions[sid])
          for sid in reference_episodes]
    n=len(rows)
    total_t=sum(row["reference_boundaries"] for row in rows)
    return {"sr":math.fsum(row["success"] for row in rows)/n,
            "ate":math.sqrt(math.fsum(row["ate_squared_sum"] for row in rows)/total_t),
            "rpe":math.sqrt(math.fsum(row["rpe_squared_sum"] for row in rows)/total_t),
            "D":math.fsum(row["D"] for row in rows)/n,
            "H":math.fsum(row["H"] for row in rows)/n,
            "DH":math.fsum(row["D"]*row["H"] for row in rows)/n,
            "score_nested_local":math.fsum(row["score_points"] for row in rows)/n}





def _poses(actions, reference):
    """Decode each [dx, dy, dyaw] in the dataset's global planar axes."""
    return compose(reference["initial_pose"], actions)


def success_rate_episode(actions, reference):
    """The episode's SR bit; final distance is strictly below its radius."""
    xy, _ = _poses(actions, reference)
    radius = max(reference["avg_step_length"], 0.6)
    return int(math.dist(xy[-1], reference["goal_pose"][:2]) < radius)


def absolute_trajectory_error_episode(actions, reference):
    """Original reference-window position RMSE for one episode."""
    xy, _ = _poses(actions, reference)
    target = reference["positions"]
    count = len(target) - 1
    if count < 1:
        raise ValueError("reference must contain motion boundaries")
    last = len(xy) - 1
    squared = (math.dist(xy[min(j, last)], target[j]) ** 2
               for j in range(1, count + 1))
    return math.sqrt(math.fsum(squared) / count)


def relative_pose_error_episode(actions, reference):
    """Original reference-window translation-increment RMSE for one episode."""
    xy, _ = _poses(actions, reference)
    target = reference["positions"]
    count = len(target) - 1
    if count < 1:
        raise ValueError("reference must contain motion boundaries")
    last = len(xy) - 1
    terms = []
    for j in range(1, count + 1):
        now, before = xy[min(j, last)], xy[min(j - 1, last)]
        ref_now, ref_before = target[j], target[j - 1]
        ex = (now[0] - before[0]) - (ref_now[0] - ref_before[0])
        ey = (now[1] - before[1]) - (ref_now[1] - ref_before[1])
        terms.append(ex * ex + ey * ey)
    # Rotating both translation errors into the reference local frame leaves
    # their Euclidean norm unchanged.
    return math.sqrt(math.fsum(terms) / count)


def full_route_fidelity(actions, reference):
    """D: full ordered route agreement and excess-distance penalty.

    For a moving reference, D = max(0, 1-F/rho) min(1, L_ref/L_pred),
    where F is continuous polygonal Frechet distance and rho is the maximum
    reference distance from its start. The core handles stationary references.
    """
    xy, yaw = _poses(actions, reference)
    return route_facing(reference["positions"], reference["yaws"], xy, yaw)[0]


def heading_and_turn_fidelity(actions, reference):
    """H: (1 - movement-facing error) / (1 + turn-process error)."""
    xy, yaw = _poses(actions, reference)
    return route_facing(reference["positions"], reference["yaws"], xy, yaw)[1]


def score_episode(actions, reference):
    """Return the five measures and this episode's contribution to Score."""
    s = success_rate_episode(actions, reference)
    a = absolute_trajectory_error_episode(actions, reference)
    r = relative_pose_error_episode(actions, reference)
    d, h = route_facing(
        reference["positions"], reference["yaws"], *_poses(actions, reference)
    )
    points = 100 * d * (1 + h) / 2 * (
        0.70 + 0.10 * s + 0.10 / (1 + a) + 0.10 / (1 + r)
    )
    return {"SR_bit": s, "ATE_episode": a, "RPE_episode": r,
            "D": d, "H": h, "Score_episode": points}


if __name__ == "__main__":
    # Entirely synthetic example. A real submission contains only actions;
    # the scorer holds reference poses and goals privately.
    toy = {
        "initial_pose": [0.0, 0.0, 0.0],
        "positions": [(j / 10, 0.0) for j in range(11)],
        "yaws": [0.0] * 11,
        "goal_pose": [1.0, 0.0, 0.0],
        "avg_step_length": 0.1,
    }
    action_sequence = [[0.1, 0.0, 0.0] for _ in range(10)]
    for name, value in score_episode(action_sequence, toy).items():
        print(f"{name}: {value:.6f}")
