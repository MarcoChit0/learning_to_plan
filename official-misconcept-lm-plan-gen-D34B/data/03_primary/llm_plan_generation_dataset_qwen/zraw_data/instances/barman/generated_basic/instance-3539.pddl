(define (problem prob)
 (:domain barman)
 (:objects 
      shaker13 - shaker
      left right - hand
      shot386 - shot
      ingredient427 ingredient299 ingredient451 - ingredient
      cocktail292 - cocktail
      dispenser140 dispenser149 dispenser100 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker13)
  (ontable shot386)
  (dispenses dispenser140 ingredient427)
  (dispenses dispenser149 ingredient299)
  (dispenses dispenser100 ingredient451)
  (clean shaker13)
  (clean shot386)
  (empty shaker13)
  (empty shot386)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker13 l0)
  (shaker-level shaker13 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail292 ingredient451)
  (cocktail-part2 cocktail292 ingredient299)
)
 (:goal
  (and
      (contains shot386 cocktail292)
)))
