;Header and description

(define (domain polycraft_generated)

;remove requirements that are not needed
(:requirements :typing :strips :fluents :negative-preconditions :equality)

(:types ;todo: enumerate types and their hierarchy here, e.g. car truck bus - vehicle
    pickaxe_breakable - breakable
    hand_breakable - pickaxe_breakable
    breakable - placeable
    placeable - physobj
    physobj - physical
    actor - physobj
    trader - actor
    pogoist - actor
    agent - actor
    oak_log - log
    distance - var
;{{object_types}}
)

(:constants 
    air - physobj
    one - distance
    two - distance
    rubber - physobj
    blue_key - physobj
)

(:predicates ;todo: define predicates here
    (holding ?v0 - physobj)
    (floating ?v0 - physobj)
    (facing_obj ?v0 - physobj ?d - distance)
    (next_to ?v0 - physobj ?v1 - physobj)
)


(:functions ;todo: define numeric functions here
    (world ?v0 - physobj)
    (inventory ?v0 - physobj)
    (container ?v0 - physobj ?v1 - physobj)
)

; define actions here
(:action approach
    :parameters    (?physobj01 - physobj ?physobj02 - physobj )
    :precondition  (and
        (>= ( world ?physobj02) 1)
        (facing_obj ?physobj01 one)
    )
    :effect  (and
        (facing_obj ?physobj02 one)
        (not (facing_obj ?physobj01 one))
    )
)

(:action approach_actor
    :parameters    (?physobj01 - physobj ?physobj02 - actor )
    :precondition  (and
        (facing_obj ?physobj01 one)
    )
    :effect  (and
        (facing_obj ?physobj02 one)
        (not (facing_obj ?physobj01 one))
    )
)

(:action break
    :parameters    (?physobj - hand_breakable)
    :precondition  (and
        (facing_obj ?physobj one)
        (not (floating ?physobj))
    )
    :effect  (and
        (facing_obj air one)
        (not (facing_obj ?physobj one))
        (increase ( inventory ?physobj) 1)
        (increase ( world air) 1)
        (decrease ( world ?physobj) 1)
    )
)


(:action break_holding_iron_pickaxe
    :parameters    (?physobj - pickaxe_breakable ?iron_pickaxe - iron_pickaxe)
    :precondition  (and
        (facing_obj ?physobj one)
        (not (floating ?physobj))
        (holding ?iron_pickaxe)
    )
    :effect  (and
        (facing_obj air one)
        (not (facing_obj ?physobj one))
        (increase ( inventory ?physobj) 1)
        (increase ( world air) 1)
        (decrease ( world ?physobj) 1)
    )
)

(:action break_diamond_ore
    :parameters    (?iron_pickaxe - iron_pickaxe)
    :precondition  (and
        (facing_obj diamond_ore one)
        (not (floating diamond_ore))
        (holding ?iron_pickaxe)
    )
    :effect  (and
        (facing_obj air one)
        (not (facing_obj diamond_ore one))
        (increase ( inventory diamond) 9)
        (increase ( world air) 1)
        (decrease ( world diamond_ore) 1)
    )
)

(:action select
    :parameters    (?prev_holding - physobj ?obj_to_select - physobj)
    :precondition  (and
        (>= ( inventory ?obj_to_select) 1)
        (holding ?prev_holding)
        (not (= ?obj_to_select air))
    )
    :effect  (and
        (holding ?obj_to_select)
        (not (holding ?prev_holding))
    )
)

(:action deselect_item
    :parameters    (?physobj01 - physobj)
    :precondition  (and
        (holding ?physobj01)
        (not (holding air))
    )
    :effect  (and
        (not (holding ?physobj01))
        (holding air)
    )
)

(:action place_sapling
    :parameters (?sapling - sapling ?log - log)
    :precondition (and
        (facing_obj air one)
        (holding ?sapling)
    )
    :effect (and
        (facing_obj ?log one)
        (not (facing_obj air one))
        (increase ( world ?log) 1)
        (decrease ( inventory ?log) 1)
    )
)

(:action place
    :parameters   (?physobj01 - placeable)
    :precondition (and
        (facing_obj air one)
        (holding ?physobj01)
    )
    :effect (and 
        (facing_obj ?physobj01 one)
        (not (facing_obj air one))
        (increase ( world ?physobj01) 1)
        (decrease ( inventory ?physobj01) 1)
    )
)

(:action collect_from_tree_tap
    :parameters (?actor - actor ?log - log)

    :precondition (and
        (holding tree_tap)
        (facing_obj ?log one)
    )
    :effect (and
        (increase ( inventory rubber) 1)
    )
)

;; not working
(:action collect_from_safe
    :parameters (?actor - actor ?safe - safe)

    :precondition (and
        (facing_obj ?safe one)
        (holding blue_key)
        (>= (container ?safe diamond) 18)
    )
    :effect (and
        (decrease (container ?safe diamond) 18)
        (increase (inventory diamond) 18)
    )
)

;; not working
(:action collect_from_chest
    :parameters (?actor - actor ?chest - plastic_chest)
    :precondition (and 
        (facing_obj ?chest one)
        (>= (container ?chest blue_key) 1)
    )
    :effect (and
        (increase (inventory blue_key) 1)
        (decrease (container ?chest blue_key) 1)
    )
)


; TODO collect from safe
; (:action collect_from_safe
;     :parameters (?actor - actor ?safe - safe)

;     :precondition (and
;         (facing_obj_obj ?actor ?safe one)
;         (holding key)
;         (= (container ?safe diamond) 18)
;     )
;     :effect (and
;         (decrease (container ?safe diamond) 18)
;         (increase (inventory ?actor diamond) 18)
;     )
; )

; additional actions, including craft and trade
;{{additional_actions}}
)
