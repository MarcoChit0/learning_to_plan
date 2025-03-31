(define (problem prob)
 (:domain barman)
 (:objects 
      shaker348 - shaker
      left right - hand
      shot235 - shot
      ingredient242 ingredient227 ingredient218 - ingredient
      cocktail1 - cocktail
      dispenser217 dispenser293 dispenser51 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker348)
  (ontable shot235)
  (dispenses dispenser217 ingredient242)
  (dispenses dispenser293 ingredient227)
  (dispenses dispenser51 ingredient218)
  (clean shaker348)
  (clean shot235)
  (empty shaker348)
  (empty shot235)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker348 l0)
  (shaker-level shaker348 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient218)
  (cocktail-part2 cocktail1 ingredient227)
)
 (:goal
  (and
      (contains shot235 cocktail1)
)))
