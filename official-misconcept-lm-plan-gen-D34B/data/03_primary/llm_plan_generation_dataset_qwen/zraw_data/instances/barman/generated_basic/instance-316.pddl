(define (problem prob)
 (:domain barman)
 (:objects 
      shaker312 - shaker
      left right - hand
      shot28 shot493 - shot
      ingredient336 ingredient473 - ingredient
      cocktail1 - cocktail
      dispenser19 dispenser397 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker312)
  (ontable shot28)
  (ontable shot493)
  (dispenses dispenser19 ingredient336)
  (dispenses dispenser397 ingredient473)
  (clean shaker312)
  (clean shot28)
  (clean shot493)
  (empty shaker312)
  (empty shot28)
  (empty shot493)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker312 l0)
  (shaker-level shaker312 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient336)
  (cocktail-part2 cocktail1 ingredient473)
)
 (:goal
  (and
      (contains shot28 cocktail1)
)))
