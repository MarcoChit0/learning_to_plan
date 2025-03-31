(define (problem prob)
 (:domain barman)
 (:objects 
      shaker184 - shaker
      left right - hand
      shot180 - shot
      ingredient300 ingredient195 - ingredient
      cocktail1 - cocktail
      dispenser366 dispenser336 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker184)
  (ontable shot180)
  (dispenses dispenser366 ingredient300)
  (dispenses dispenser336 ingredient195)
  (clean shaker184)
  (clean shot180)
  (empty shaker184)
  (empty shot180)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker184 l0)
  (shaker-level shaker184 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient195)
  (cocktail-part2 cocktail1 ingredient300)
)
 (:goal
  (and
      (contains shot180 cocktail1)
)))
