(define (problem prob)
 (:domain barman)
 (:objects 
      shaker101 - shaker
      left right - hand
      shot149 shot335 shot449 - shot
      ingredient369 ingredient228 ingredient495 ingredient128 - ingredient
      cocktail129 - cocktail
      dispenser344 dispenser477 dispenser25 dispenser306 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker101)
  (ontable shot149)
  (ontable shot335)
  (ontable shot449)
  (dispenses dispenser344 ingredient369)
  (dispenses dispenser477 ingredient228)
  (dispenses dispenser25 ingredient495)
  (dispenses dispenser306 ingredient128)
  (clean shaker101)
  (clean shot149)
  (clean shot335)
  (clean shot449)
  (empty shaker101)
  (empty shot149)
  (empty shot335)
  (empty shot449)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker101 l0)
  (shaker-level shaker101 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail129 ingredient128)
  (cocktail-part2 cocktail129 ingredient369)
)
 (:goal
  (and
      (contains shot149 cocktail129)
      (contains shot335 ingredient495)
)))
