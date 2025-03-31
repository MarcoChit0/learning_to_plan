(define (problem prob)
 (:domain barman)
 (:objects 
      shaker75 - shaker
      left right - hand
      shot417 shot39 shot32 - shot
      ingredient349 ingredient2 - ingredient
      cocktail492 - cocktail
      dispenser419 dispenser131 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker75)
  (ontable shot417)
  (ontable shot39)
  (ontable shot32)
  (dispenses dispenser419 ingredient349)
  (dispenses dispenser131 ingredient2)
  (clean shaker75)
  (clean shot417)
  (clean shot39)
  (clean shot32)
  (empty shaker75)
  (empty shot417)
  (empty shot39)
  (empty shot32)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker75 l0)
  (shaker-level shaker75 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail492 ingredient349)
  (cocktail-part2 cocktail492 ingredient2)
)
 (:goal
  (and
      (contains shot417 cocktail492)
      (contains shot39 cocktail492)
)))
