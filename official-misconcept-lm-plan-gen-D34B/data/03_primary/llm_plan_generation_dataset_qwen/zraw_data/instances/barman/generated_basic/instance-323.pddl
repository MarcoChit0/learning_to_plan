(define (problem prob)
 (:domain barman)
 (:objects 
      shaker397 - shaker
      left right - hand
      shot100 - shot
      ingredient23 ingredient160 - ingredient
      cocktail1 - cocktail
      dispenser141 dispenser140 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker397)
  (ontable shot100)
  (dispenses dispenser141 ingredient23)
  (dispenses dispenser140 ingredient160)
  (clean shaker397)
  (clean shot100)
  (empty shaker397)
  (empty shot100)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker397 l0)
  (shaker-level shaker397 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient160)
  (cocktail-part2 cocktail1 ingredient23)
)
 (:goal
  (and
      (contains shot100 cocktail1)
)))
