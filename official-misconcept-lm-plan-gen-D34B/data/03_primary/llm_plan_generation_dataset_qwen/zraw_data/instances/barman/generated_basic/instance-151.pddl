(define (problem prob)
 (:domain barman)
 (:objects 
      shaker422 - shaker
      left right - hand
      shot486 - shot
      ingredient299 ingredient479 ingredient123 ingredient1 - ingredient
      cocktail1 - cocktail
      dispenser185 dispenser237 dispenser47 dispenser53 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker422)
  (ontable shot486)
  (dispenses dispenser185 ingredient299)
  (dispenses dispenser237 ingredient479)
  (dispenses dispenser47 ingredient123)
  (dispenses dispenser53 ingredient1)
  (clean shaker422)
  (clean shot486)
  (empty shaker422)
  (empty shot486)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker422 l0)
  (shaker-level shaker422 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient299)
  (cocktail-part2 cocktail1 ingredient1)
)
 (:goal
  (and
      (contains shot486 cocktail1)
)))
