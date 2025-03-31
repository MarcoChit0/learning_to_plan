(define (problem prob)
 (:domain barman)
 (:objects 
      shaker311 - shaker
      left right - hand
      shot415 - shot
      ingredient180 ingredient284 ingredient33 - ingredient
      cocktail246 - cocktail
      dispenser130 dispenser284 dispenser92 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker311)
  (ontable shot415)
  (dispenses dispenser130 ingredient180)
  (dispenses dispenser284 ingredient284)
  (dispenses dispenser92 ingredient33)
  (clean shaker311)
  (clean shot415)
  (empty shaker311)
  (empty shot415)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker311 l0)
  (shaker-level shaker311 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail246 ingredient284)
  (cocktail-part2 cocktail246 ingredient180)
)
 (:goal
  (and
      (contains shot415 cocktail246)
)))
