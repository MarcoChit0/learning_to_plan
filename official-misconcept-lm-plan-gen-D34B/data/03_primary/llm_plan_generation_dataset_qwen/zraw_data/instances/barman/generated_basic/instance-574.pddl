(define (problem prob)
 (:domain barman)
 (:objects 
      shaker114 - shaker
      left right - hand
      shot133 shot192 - shot
      ingredient128 ingredient499 - ingredient
      cocktail1 - cocktail
      dispenser243 dispenser309 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker114)
  (ontable shot133)
  (ontable shot192)
  (dispenses dispenser243 ingredient128)
  (dispenses dispenser309 ingredient499)
  (clean shaker114)
  (clean shot133)
  (clean shot192)
  (empty shaker114)
  (empty shot133)
  (empty shot192)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker114 l0)
  (shaker-level shaker114 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient128)
  (cocktail-part2 cocktail1 ingredient499)
)
 (:goal
  (and
      (contains shot133 cocktail1)
)))
