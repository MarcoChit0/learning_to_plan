(define (problem prob)
 (:domain barman)
 (:objects 
      shaker322 - shaker
      left right - hand
      shot415 - shot
      ingredient290 ingredient371 ingredient262 - ingredient
      cocktail32 - cocktail
      dispenser334 dispenser367 dispenser16 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker322)
  (ontable shot415)
  (dispenses dispenser334 ingredient290)
  (dispenses dispenser367 ingredient371)
  (dispenses dispenser16 ingredient262)
  (clean shaker322)
  (clean shot415)
  (empty shaker322)
  (empty shot415)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker322 l0)
  (shaker-level shaker322 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail32 ingredient290)
  (cocktail-part2 cocktail32 ingredient371)
)
 (:goal
  (and
      (contains shot415 cocktail32)
)))
