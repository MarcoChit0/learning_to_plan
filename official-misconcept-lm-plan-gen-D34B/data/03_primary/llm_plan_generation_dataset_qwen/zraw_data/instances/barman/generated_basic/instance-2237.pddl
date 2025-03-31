(define (problem prob)
 (:domain barman)
 (:objects 
      shaker125 - shaker
      left right - hand
      shot147 - shot
      ingredient33 ingredient456 - ingredient
      cocktail126 - cocktail
      dispenser114 dispenser399 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker125)
  (ontable shot147)
  (dispenses dispenser114 ingredient33)
  (dispenses dispenser399 ingredient456)
  (clean shaker125)
  (clean shot147)
  (empty shaker125)
  (empty shot147)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker125 l0)
  (shaker-level shaker125 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail126 ingredient456)
  (cocktail-part2 cocktail126 ingredient33)
)
 (:goal
  (and
      (contains shot147 cocktail126)
)))
