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
    bedrock - placeable
    door - placeable
    safe - placeable
    plastic_chest - placeable
    tree_tap - placeable
    oak_log - hand_breakable
    diamond_ore - pickaxe_breakable
    iron_pickaxe - physobj
    crafting_table - hand_breakable
    block_of_platinum - pickaxe_breakable
    block_of_titanium - placeable
    sapling - placeable
    planks - physobj
    stick - physobj
    diamond - physobj
    block_of_diamond - physobj
    rubber - physobj
    pogo_stick - physobj
    blue_key - physobj
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
    (facing ?v0 - physobj ?d - distance)
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
        (facing ?physobj01 one)
    )
    :effect  (and
        (facing ?physobj02 one)
        (not (facing ?physobj01 one))
    )
)

(:action approach_actor
    :parameters    (?physobj01 - physobj ?physobj02 - actor )
    :precondition  (and
        (facing ?physobj01 one)
    )
    :effect  (and
        (facing ?physobj02 one)
        (not (facing ?physobj01 one))
    )
)

(:action break
    :parameters    (?physobj - hand_breakable)
    :precondition  (and
        (facing ?physobj one)
        (not (floating ?physobj))
    )
    :effect  (and
        (facing air one)
        (not (facing ?physobj one))
        (increase ( inventory ?physobj) 1)
        (increase ( world air) 1)
        (decrease ( world ?physobj) 1)
    )
)


(:action break_holding_iron_pickaxe
    :parameters    (?physobj - pickaxe_breakable ?iron_pickaxe - iron_pickaxe)
    :precondition  (and
        (facing ?physobj one)
        (not (floating ?physobj))
        (holding ?iron_pickaxe)
    )
    :effect  (and
        (facing air one)
        (not (facing ?physobj one))
        (increase ( inventory ?physobj) 1)
        (increase ( world air) 1)
        (decrease ( world ?physobj) 1)
    )
)

(:action break_diamond_ore
    :parameters    (?iron_pickaxe - iron_pickaxe)
    :precondition  (and
        (facing diamond_ore one)
        (not (floating diamond_ore))
        (holding ?iron_pickaxe)
    )
    :effect  (and
        (facing air one)
        (not (facing diamond_ore one))
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

(:action place
    :parameters   (?physobj01 - placeable)
    :precondition (and
        (facing air one)
        (holding ?physobj01)
    )
    :effect (and 
        (facing ?physobj01 one)
        (not (facing air one))
        (increase ( world ?physobj01) 1)
        (decrease ( inventory ?physobj01) 1)
    )
)

(:action collect_from_tree_tap
    :parameters (?actor - actor ?log - log)

    :precondition (and
        (holding tree_tap)
        (facing ?log one)
    )
    :effect (and
        (increase ( inventory rubber) 1)
    )
)

;; not working
(:action collect_from_safe
    :parameters (?actor - actor ?safe - safe)

    :precondition (and
        (facing ?safe one)
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
        (facing ?chest one)
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
;         (facing_obj ?actor ?safe one)
;         (holding key)
;         (= (container ?safe diamond) 18)
;     )
;     :effect (and
;         (decrease (container ?safe diamond) 18)
;         (increase (inventory ?actor diamond) 18)
;     )
; )

; additional actions, including craft and trade
(:action craft_stick
    :parameters ()
    :precondition (and
        (>= ( inventory planks) 2)
    )
    :effect (and
        (decrease ( inventory planks) 2)
        (increase ( inventory stick) 4)
    )
)


(:action craft_planks
    :parameters ()
    :precondition (and
        (>= ( inventory oak_log) 1)
    )
    :effect (and
        (decrease ( inventory oak_log) 1)
        (increase ( inventory planks) 4)
    )
)


(:action craft_block_of_diamond
    :parameters ()
    :precondition (and
        (facing crafting_table one)
        (>= ( inventory diamond) 9)
    )
    :effect (and
        (decrease ( inventory diamond) 9)
        (increase ( inventory block_of_diamond) 1)
    )
)


(:action craft_tree_tap
    :parameters ()
    :precondition (and
        (facing crafting_table one)
        (>= ( inventory planks) 5)
        (>= ( inventory stick) 1)
    )
    :effect (and
        (decrease ( inventory planks) 5)
        (decrease ( inventory stick) 1)
        (increase ( inventory tree_tap) 1)
    )
)


(:action craft_pogo_stick
    :parameters ()
    :precondition (and
        (facing crafting_table one)
        (>= ( inventory stick) 2)
        (>= ( inventory block_of_titanium) 2)
        (>= ( inventory block_of_diamond) 2)
        (>= ( inventory rubber) 1)
    )
    :effect (and
        (decrease ( inventory stick) 2)
        (decrease ( inventory block_of_titanium) 2)
        (decrease ( inventory block_of_diamond) 2)
        (decrease ( inventory rubber) 1)
        (increase ( inventory pogo_stick) 1)
    )
)


(:action trade_block_of_titanium_1
    :parameters ()
    :precondition (and
        (facing entity_103 one)
        (>= ( inventory block_of_platinum) 1)
    )
    :effect (and
        (decrease ( inventory block_of_platinum) 1)
        (increase ( inventory block_of_titanium) 1)
    )
)


(:action trade_block_of_platinum_1
    :parameters ()
    :precondition (and
        (facing entity_103 one)
        (>= ( inventory diamond) 18)
    )
    :effect (and
        (decrease ( inventory diamond) 18)
        (increase ( inventory block_of_platinum) 1)
    )
)


(:action trade_diamond_1
    :parameters ()
    :precondition (and
        (facing entity_104 one)
        (>= ( inventory block_of_platinum) 2)
    )
    :effect (and
        (decrease ( inventory block_of_platinum) 2)
        (increase ( inventory diamond) 9)
    )
)


(:action trade_block_of_titanium_2
    :parameters ()
    :precondition (and
        (facing entity_104 one)
        (>= ( inventory oak_log) 10)
    )
    :effect (and
        (decrease ( inventory oak_log) 10)
        (increase ( inventory block_of_titanium) 2)
    )
)

)
