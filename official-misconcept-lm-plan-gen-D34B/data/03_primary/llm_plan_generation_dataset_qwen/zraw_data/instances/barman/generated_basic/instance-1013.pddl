(define (problem prob)
 (:domain barman)
 (:objects 
      shaker347 - shaker
      left right - hand
      shot106 - shot
      ingredient416 ingredient405 ingredient55 - ingredient
      cocktail1 - cocktail
      dispenser387 dispenser419 dispenser100 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker347)
  (ontable shot106)
  (dispenses dispenser387 ingredient416)
  (dispenses dispenser419 ingredient405)
  (dispenses dispenser100 ingredient55)
  (clean shaker347)
  (clean shot106)
  (empty shaker347)
  (empty shot106)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker347 l0)
  (shaker-level shaker347 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient405)
  (cocktail-part2 cocktail1 ingredient416)
)
 (:goal
  (and
      (contains shot106 cocktail1)
)))
