(define
	(problem polycraft_problem)
	(:domain polycraft_generated)
    (:objects 
        bedrock - bedrock
        door - door
        safe - safe
        plastic_chest - plastic_chest
        tree_tap - tree_tap
        oak_log - oak_log
        diamond_ore - diamond_ore
        iron_pickaxe - iron_pickaxe
        crafting_table - crafting_table
        block_of_platinum - block_of_platinum
        block_of_titanium - block_of_titanium
        sapling - sapling
        planks - planks
        stick - stick
        diamond - diamond
        block_of_diamond - block_of_diamond
        rubber - rubber
        pogo_stick - pogo_stick
        blue_key - blue_key
        entity_1 - agent
        entity_103 - trader
        entity_104 - trader
        entity_102 - pogoist
    )

    (:init
        (= (world air) 1004)
        (= (world bedrock) 482)
        (= (world oak_log) 5)
        (= (world crafting_table) 1)
        (= (world diamond_ore) 4)
        (= (world door) 2)
        (= (world safe) 1)
        (= (world plastic_chest) 1)
        (= (world block_of_platinum) 4)
        (= (inventory iron_pickaxe) 1)
        (= (inventory bedrock) 0)
        (= (inventory door) 0)
        (= (inventory safe) 0)
        (= (inventory plastic_chest) 0)
        (= (inventory tree_tap) 0)
        (= (inventory oak_log) 0)
        (= (inventory diamond_ore) 0)
        (= (inventory crafting_table) 0)
        (= (inventory block_of_platinum) 0)
        (= (inventory block_of_titanium) 0)
        (= (inventory sapling) 0)
        (= (inventory planks) 0)
        (= (inventory stick) 0)
        (= (inventory diamond) 0)
        (= (inventory block_of_diamond) 0)
        (= (inventory rubber) 0)
        (= (inventory pogo_stick) 0)
        (= (inventory blue_key) 0)
        (facing air one)
        (holding air)
    )

    (:goal (>= (inventory pogo_stick) 1))
)
