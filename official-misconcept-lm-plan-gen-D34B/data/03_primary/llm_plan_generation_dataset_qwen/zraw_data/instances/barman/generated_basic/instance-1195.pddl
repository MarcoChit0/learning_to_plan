(define (problem prob)
 (:domain barman)
 (:objects 
      shaker369 - shaker
      left right - hand
      shot403 shot423 - shot
      ingredient263 ingredient172 - ingredient
      cocktail334 - cocktail
      dispenser86 dispenser468 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker369)
  (ontable shot403)
  (ontable shot423)
  (dispenses dispenser86 ingredient263)
  (dispenses dispenser468 ingredient172)
  (clean shaker369)
  (clean shot403)
  (clean shot423)
  (empty shaker369)
  (empty shot403)
  (empty shot423)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker369 l0)
  (shaker-level shaker369 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail334 ingredient172)
  (cocktail-part2 cocktail334 ingredient263)
)
 (:goal
  (and
      (contains shot403 cocktail334)
)))
