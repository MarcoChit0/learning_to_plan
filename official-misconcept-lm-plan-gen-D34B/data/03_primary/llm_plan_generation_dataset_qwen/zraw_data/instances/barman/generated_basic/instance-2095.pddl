(define (problem prob)
 (:domain barman)
 (:objects 
      shaker297 - shaker
      left right - hand
      shot31 shot125 - shot
      ingredient332 ingredient395 - ingredient
      cocktail457 - cocktail
      dispenser397 dispenser94 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker297)
  (ontable shot31)
  (ontable shot125)
  (dispenses dispenser397 ingredient332)
  (dispenses dispenser94 ingredient395)
  (clean shaker297)
  (clean shot31)
  (clean shot125)
  (empty shaker297)
  (empty shot31)
  (empty shot125)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker297 l0)
  (shaker-level shaker297 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail457 ingredient332)
  (cocktail-part2 cocktail457 ingredient395)
)
 (:goal
  (and
      (contains shot31 cocktail457)
)))
