(define (problem prob)
 (:domain barman)
 (:objects 
      shaker451 - shaker
      left right - hand
      shot374 shot335 shot59 - shot
      ingredient182 ingredient140 - ingredient
      cocktail48 - cocktail
      dispenser359 dispenser363 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker451)
  (ontable shot374)
  (ontable shot335)
  (ontable shot59)
  (dispenses dispenser359 ingredient182)
  (dispenses dispenser363 ingredient140)
  (clean shaker451)
  (clean shot374)
  (clean shot335)
  (clean shot59)
  (empty shaker451)
  (empty shot374)
  (empty shot335)
  (empty shot59)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker451 l0)
  (shaker-level shaker451 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail48 ingredient140)
  (cocktail-part2 cocktail48 ingredient182)
)
 (:goal
  (and
      (contains shot374 cocktail48)
      (contains shot335 cocktail48)
)))
