(define (problem prob)
 (:domain barman)
 (:objects 
      shaker490 - shaker
      left right - hand
      shot95 shot277 - shot
      ingredient293 ingredient492 ingredient170 - ingredient
      cocktail1 - cocktail
      dispenser185 dispenser438 dispenser130 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker490)
  (ontable shot95)
  (ontable shot277)
  (dispenses dispenser185 ingredient293)
  (dispenses dispenser438 ingredient492)
  (dispenses dispenser130 ingredient170)
  (clean shaker490)
  (clean shot95)
  (clean shot277)
  (empty shaker490)
  (empty shot95)
  (empty shot277)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker490 l0)
  (shaker-level shaker490 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient492)
  (cocktail-part2 cocktail1 ingredient293)
)
 (:goal
  (and
      (contains shot95 cocktail1)
)))
