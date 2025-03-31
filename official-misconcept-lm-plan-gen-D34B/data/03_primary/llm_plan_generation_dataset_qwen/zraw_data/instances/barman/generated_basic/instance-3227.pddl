(define (problem prob)
 (:domain barman)
 (:objects 
      shaker417 - shaker
      left right - hand
      shot385 - shot
      ingredient120 ingredient357 - ingredient
      cocktail499 - cocktail
      dispenser302 dispenser339 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker417)
  (ontable shot385)
  (dispenses dispenser302 ingredient120)
  (dispenses dispenser339 ingredient357)
  (clean shaker417)
  (clean shot385)
  (empty shaker417)
  (empty shot385)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker417 l0)
  (shaker-level shaker417 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail499 ingredient120)
  (cocktail-part2 cocktail499 ingredient357)
)
 (:goal
  (and
      (contains shot385 cocktail499)
)))
