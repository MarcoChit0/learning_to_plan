(define (problem prob)
 (:domain barman)
 (:objects 
      shaker212 - shaker
      left right - hand
      shot358 shot58 - shot
      ingredient95 ingredient495 - ingredient
      cocktail307 - cocktail
      dispenser496 dispenser390 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker212)
  (ontable shot358)
  (ontable shot58)
  (dispenses dispenser496 ingredient95)
  (dispenses dispenser390 ingredient495)
  (clean shaker212)
  (clean shot358)
  (clean shot58)
  (empty shaker212)
  (empty shot358)
  (empty shot58)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker212 l0)
  (shaker-level shaker212 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail307 ingredient495)
  (cocktail-part2 cocktail307 ingredient95)
)
 (:goal
  (and
      (contains shot358 cocktail307)
)))
