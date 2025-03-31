(define (problem prob)
 (:domain barman)
 (:objects 
      shaker310 - shaker
      left right - hand
      shot242 - shot
      ingredient378 ingredient396 ingredient12 - ingredient
      cocktail459 - cocktail
      dispenser42 dispenser491 dispenser453 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker310)
  (ontable shot242)
  (dispenses dispenser42 ingredient378)
  (dispenses dispenser491 ingredient396)
  (dispenses dispenser453 ingredient12)
  (clean shaker310)
  (clean shot242)
  (empty shaker310)
  (empty shot242)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker310 l0)
  (shaker-level shaker310 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail459 ingredient396)
  (cocktail-part2 cocktail459 ingredient378)
)
 (:goal
  (and
      (contains shot242 cocktail459)
)))
