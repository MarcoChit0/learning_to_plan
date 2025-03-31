(define (problem prob)
 (:domain barman)
 (:objects 
      shaker421 - shaker
      left right - hand
      shot358 - shot
      ingredient300 ingredient442 ingredient147 - ingredient
      cocktail208 - cocktail
      dispenser474 dispenser86 dispenser451 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker421)
  (ontable shot358)
  (dispenses dispenser474 ingredient300)
  (dispenses dispenser86 ingredient442)
  (dispenses dispenser451 ingredient147)
  (clean shaker421)
  (clean shot358)
  (empty shaker421)
  (empty shot358)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker421 l0)
  (shaker-level shaker421 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail208 ingredient147)
  (cocktail-part2 cocktail208 ingredient300)
)
 (:goal
  (and
      (contains shot358 cocktail208)
)))
