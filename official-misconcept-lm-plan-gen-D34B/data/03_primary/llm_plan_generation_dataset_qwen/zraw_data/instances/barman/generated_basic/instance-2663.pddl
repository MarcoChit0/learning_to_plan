(define (problem prob)
 (:domain barman)
 (:objects 
      shaker172 - shaker
      left right - hand
      shot96 - shot
      ingredient479 ingredient61 ingredient486 ingredient384 - ingredient
      cocktail350 - cocktail
      dispenser81 dispenser482 dispenser5 dispenser125 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker172)
  (ontable shot96)
  (dispenses dispenser81 ingredient479)
  (dispenses dispenser482 ingredient61)
  (dispenses dispenser5 ingredient486)
  (dispenses dispenser125 ingredient384)
  (clean shaker172)
  (clean shot96)
  (empty shaker172)
  (empty shot96)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker172 l0)
  (shaker-level shaker172 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail350 ingredient384)
  (cocktail-part2 cocktail350 ingredient61)
)
 (:goal
  (and
      (contains shot96 cocktail350)
)))
