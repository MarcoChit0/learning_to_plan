(define (problem prob)
 (:domain barman)
 (:objects 
      shaker461 - shaker
      left right - hand
      shot108 shot54 - shot
      ingredient459 ingredient32 ingredient352 - ingredient
      cocktail331 - cocktail
      dispenser400 dispenser434 dispenser369 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker461)
  (ontable shot108)
  (ontable shot54)
  (dispenses dispenser400 ingredient459)
  (dispenses dispenser434 ingredient32)
  (dispenses dispenser369 ingredient352)
  (clean shaker461)
  (clean shot108)
  (clean shot54)
  (empty shaker461)
  (empty shot108)
  (empty shot54)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker461 l0)
  (shaker-level shaker461 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail331 ingredient352)
  (cocktail-part2 cocktail331 ingredient459)
)
 (:goal
  (and
      (contains shot108 cocktail331)
)))
