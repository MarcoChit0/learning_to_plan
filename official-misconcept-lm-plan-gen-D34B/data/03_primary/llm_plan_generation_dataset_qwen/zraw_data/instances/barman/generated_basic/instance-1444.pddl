(define (problem prob)
 (:domain barman)
 (:objects 
      shaker184 - shaker
      left right - hand
      shot108 - shot
      ingredient60 ingredient119 ingredient411 - ingredient
      cocktail491 - cocktail
      dispenser90 dispenser287 dispenser372 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker184)
  (ontable shot108)
  (dispenses dispenser90 ingredient60)
  (dispenses dispenser287 ingredient119)
  (dispenses dispenser372 ingredient411)
  (clean shaker184)
  (clean shot108)
  (empty shaker184)
  (empty shot108)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker184 l0)
  (shaker-level shaker184 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail491 ingredient411)
  (cocktail-part2 cocktail491 ingredient60)
)
 (:goal
  (and
      (contains shot108 cocktail491)
)))
