(define (problem prob)
 (:domain barman)
 (:objects 
      shaker215 - shaker
      left right - hand
      shot99 shot192 shot406 - shot
      ingredient72 ingredient246 ingredient182 - ingredient
      cocktail315 - cocktail
      dispenser96 dispenser327 dispenser10 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker215)
  (ontable shot99)
  (ontable shot192)
  (ontable shot406)
  (dispenses dispenser96 ingredient72)
  (dispenses dispenser327 ingredient246)
  (dispenses dispenser10 ingredient182)
  (clean shaker215)
  (clean shot99)
  (clean shot192)
  (clean shot406)
  (empty shaker215)
  (empty shot99)
  (empty shot192)
  (empty shot406)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker215 l0)
  (shaker-level shaker215 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail315 ingredient72)
  (cocktail-part2 cocktail315 ingredient182)
)
 (:goal
  (and
      (contains shot99 cocktail315)
      (contains shot192 cocktail315)
)))
