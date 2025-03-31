(define (problem prob)
 (:domain barman)
 (:objects 
      shaker490 - shaker
      left right - hand
      shot325 - shot
      ingredient414 ingredient207 ingredient375 - ingredient
      cocktail1 - cocktail
      dispenser296 dispenser340 dispenser116 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker490)
  (ontable shot325)
  (dispenses dispenser296 ingredient414)
  (dispenses dispenser340 ingredient207)
  (dispenses dispenser116 ingredient375)
  (clean shaker490)
  (clean shot325)
  (empty shaker490)
  (empty shot325)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker490 l0)
  (shaker-level shaker490 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient207)
  (cocktail-part2 cocktail1 ingredient375)
)
 (:goal
  (and
      (contains shot325 cocktail1)
)))
