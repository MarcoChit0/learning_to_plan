(define (problem prob)
 (:domain barman)
 (:objects 
      shaker81 - shaker
      left right - hand
      shot70 - shot
      ingredient59 ingredient188 - ingredient
      cocktail211 - cocktail
      dispenser44 dispenser72 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker81)
  (ontable shot70)
  (dispenses dispenser44 ingredient59)
  (dispenses dispenser72 ingredient188)
  (clean shaker81)
  (clean shot70)
  (empty shaker81)
  (empty shot70)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker81 l0)
  (shaker-level shaker81 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail211 ingredient188)
  (cocktail-part2 cocktail211 ingredient59)
)
 (:goal
  (and
      (contains shot70 cocktail211)
)))
