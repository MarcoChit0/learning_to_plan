(define (problem prob)
 (:domain barman)
 (:objects 
      shaker26 - shaker
      left right - hand
      shot242 - shot
      ingredient479 ingredient214 ingredient181 - ingredient
      cocktail1 - cocktail
      dispenser283 dispenser484 dispenser364 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker26)
  (ontable shot242)
  (dispenses dispenser283 ingredient479)
  (dispenses dispenser484 ingredient214)
  (dispenses dispenser364 ingredient181)
  (clean shaker26)
  (clean shot242)
  (empty shaker26)
  (empty shot242)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker26 l0)
  (shaker-level shaker26 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient181)
  (cocktail-part2 cocktail1 ingredient479)
)
 (:goal
  (and
      (contains shot242 cocktail1)
)))
