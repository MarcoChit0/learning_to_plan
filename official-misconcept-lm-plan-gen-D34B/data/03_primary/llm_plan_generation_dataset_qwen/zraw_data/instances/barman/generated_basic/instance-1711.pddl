(define (problem prob)
 (:domain barman)
 (:objects 
      shaker361 - shaker
      left right - hand
      shot494 - shot
      ingredient460 ingredient349 ingredient307 ingredient352 - ingredient
      cocktail177 - cocktail
      dispenser486 dispenser343 dispenser107 dispenser216 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker361)
  (ontable shot494)
  (dispenses dispenser486 ingredient460)
  (dispenses dispenser343 ingredient349)
  (dispenses dispenser107 ingredient307)
  (dispenses dispenser216 ingredient352)
  (clean shaker361)
  (clean shot494)
  (empty shaker361)
  (empty shot494)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker361 l0)
  (shaker-level shaker361 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail177 ingredient349)
  (cocktail-part2 cocktail177 ingredient460)
)
 (:goal
  (and
      (contains shot494 cocktail177)
)))
