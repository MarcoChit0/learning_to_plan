(define (problem prob)
 (:domain barman)
 (:objects 
      shaker357 - shaker
      left right - hand
      shot230 - shot
      ingredient499 ingredient189 - ingredient
      cocktail434 - cocktail
      dispenser354 dispenser23 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker357)
  (ontable shot230)
  (dispenses dispenser354 ingredient499)
  (dispenses dispenser23 ingredient189)
  (clean shaker357)
  (clean shot230)
  (empty shaker357)
  (empty shot230)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker357 l0)
  (shaker-level shaker357 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail434 ingredient499)
  (cocktail-part2 cocktail434 ingredient189)
)
 (:goal
  (and
      (contains shot230 cocktail434)
)))
