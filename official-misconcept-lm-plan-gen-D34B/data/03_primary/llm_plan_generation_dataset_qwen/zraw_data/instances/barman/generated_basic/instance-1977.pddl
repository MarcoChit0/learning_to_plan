(define (problem prob)
 (:domain barman)
 (:objects 
      shaker136 - shaker
      left right - hand
      shot470 shot465 - shot
      ingredient373 ingredient320 ingredient22 - ingredient
      cocktail170 - cocktail
      dispenser172 dispenser342 dispenser170 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker136)
  (ontable shot470)
  (ontable shot465)
  (dispenses dispenser172 ingredient373)
  (dispenses dispenser342 ingredient320)
  (dispenses dispenser170 ingredient22)
  (clean shaker136)
  (clean shot470)
  (clean shot465)
  (empty shaker136)
  (empty shot470)
  (empty shot465)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker136 l0)
  (shaker-level shaker136 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail170 ingredient320)
  (cocktail-part2 cocktail170 ingredient22)
)
 (:goal
  (and
      (contains shot470 cocktail170)
)))
