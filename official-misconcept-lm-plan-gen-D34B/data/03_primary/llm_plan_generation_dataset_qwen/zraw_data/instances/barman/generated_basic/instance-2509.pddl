(define (problem prob)
 (:domain barman)
 (:objects 
      shaker367 - shaker
      left right - hand
      shot258 - shot
      ingredient57 ingredient191 - ingredient
      cocktail291 - cocktail
      dispenser34 dispenser458 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker367)
  (ontable shot258)
  (dispenses dispenser34 ingredient57)
  (dispenses dispenser458 ingredient191)
  (clean shaker367)
  (clean shot258)
  (empty shaker367)
  (empty shot258)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker367 l0)
  (shaker-level shaker367 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail291 ingredient191)
  (cocktail-part2 cocktail291 ingredient57)
)
 (:goal
  (and
      (contains shot258 cocktail291)
)))
