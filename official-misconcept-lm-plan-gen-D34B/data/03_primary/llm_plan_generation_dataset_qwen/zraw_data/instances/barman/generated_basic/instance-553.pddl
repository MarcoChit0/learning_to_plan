(define (problem prob)
 (:domain barman)
 (:objects 
      shaker1 - shaker
      left right - hand
      shot185 - shot
      ingredient198 ingredient353 ingredient3 - ingredient
      cocktail1 - cocktail
      dispenser194 dispenser136 dispenser84 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker1)
  (ontable shot185)
  (dispenses dispenser194 ingredient198)
  (dispenses dispenser136 ingredient353)
  (dispenses dispenser84 ingredient3)
  (clean shaker1)
  (clean shot185)
  (empty shaker1)
  (empty shot185)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker1 l0)
  (shaker-level shaker1 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient3)
  (cocktail-part2 cocktail1 ingredient353)
)
 (:goal
  (and
      (contains shot185 cocktail1)
)))
