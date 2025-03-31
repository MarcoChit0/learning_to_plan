(define (problem prob)
 (:domain barman)
 (:objects 
      shaker485 - shaker
      left right - hand
      shot335 - shot
      ingredient171 ingredient496 ingredient210 - ingredient
      cocktail374 - cocktail
      dispenser380 dispenser82 dispenser383 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker485)
  (ontable shot335)
  (dispenses dispenser380 ingredient171)
  (dispenses dispenser82 ingredient496)
  (dispenses dispenser383 ingredient210)
  (clean shaker485)
  (clean shot335)
  (empty shaker485)
  (empty shot335)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker485 l0)
  (shaker-level shaker485 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail374 ingredient171)
  (cocktail-part2 cocktail374 ingredient496)
)
 (:goal
  (and
      (contains shot335 cocktail374)
)))
