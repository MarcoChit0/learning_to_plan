(define (problem prob)
 (:domain barman)
 (:objects 
      shaker337 - shaker
      left right - hand
      shot267 - shot
      ingredient263 ingredient383 - ingredient
      cocktail1 - cocktail
      dispenser25 dispenser194 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker337)
  (ontable shot267)
  (dispenses dispenser25 ingredient263)
  (dispenses dispenser194 ingredient383)
  (clean shaker337)
  (clean shot267)
  (empty shaker337)
  (empty shot267)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker337 l0)
  (shaker-level shaker337 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient263)
  (cocktail-part2 cocktail1 ingredient383)
)
 (:goal
  (and
      (contains shot267 cocktail1)
)))
