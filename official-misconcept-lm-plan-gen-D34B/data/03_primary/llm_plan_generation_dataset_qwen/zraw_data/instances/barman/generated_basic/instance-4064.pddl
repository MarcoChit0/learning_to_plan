(define (problem prob)
 (:domain barman)
 (:objects 
      shaker405 - shaker
      left right - hand
      shot409 shot419 - shot
      ingredient451 ingredient310 - ingredient
      cocktail461 - cocktail
      dispenser373 dispenser428 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker405)
  (ontable shot409)
  (ontable shot419)
  (dispenses dispenser373 ingredient451)
  (dispenses dispenser428 ingredient310)
  (clean shaker405)
  (clean shot409)
  (clean shot419)
  (empty shaker405)
  (empty shot409)
  (empty shot419)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker405 l0)
  (shaker-level shaker405 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail461 ingredient451)
  (cocktail-part2 cocktail461 ingredient310)
)
 (:goal
  (and
      (contains shot409 cocktail461)
)))
