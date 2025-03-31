(define (problem prob)
 (:domain barman)
 (:objects 
      shaker143 - shaker
      left right - hand
      shot235 - shot
      ingredient347 ingredient46 - ingredient
      cocktail1 - cocktail
      dispenser146 dispenser141 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker143)
  (ontable shot235)
  (dispenses dispenser146 ingredient347)
  (dispenses dispenser141 ingredient46)
  (clean shaker143)
  (clean shot235)
  (empty shaker143)
  (empty shot235)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker143 l0)
  (shaker-level shaker143 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient46)
  (cocktail-part2 cocktail1 ingredient347)
)
 (:goal
  (and
      (contains shot235 cocktail1)
)))
