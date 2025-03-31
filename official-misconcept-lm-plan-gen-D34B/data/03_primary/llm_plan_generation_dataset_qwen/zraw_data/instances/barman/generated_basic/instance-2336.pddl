(define (problem prob)
 (:domain barman)
 (:objects 
      shaker355 - shaker
      left right - hand
      shot242 - shot
      ingredient257 ingredient293 - ingredient
      cocktail198 - cocktail
      dispenser382 dispenser30 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker355)
  (ontable shot242)
  (dispenses dispenser382 ingredient257)
  (dispenses dispenser30 ingredient293)
  (clean shaker355)
  (clean shot242)
  (empty shaker355)
  (empty shot242)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker355 l0)
  (shaker-level shaker355 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail198 ingredient257)
  (cocktail-part2 cocktail198 ingredient293)
)
 (:goal
  (and
      (contains shot242 cocktail198)
)))
