(define (problem prob)
 (:domain barman)
 (:objects 
      shaker156 - shaker
      left right - hand
      shot197 - shot
      ingredient103 ingredient73 ingredient385 ingredient291 - ingredient
      cocktail293 - cocktail
      dispenser408 dispenser252 dispenser295 dispenser140 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker156)
  (ontable shot197)
  (dispenses dispenser408 ingredient103)
  (dispenses dispenser252 ingredient73)
  (dispenses dispenser295 ingredient385)
  (dispenses dispenser140 ingredient291)
  (clean shaker156)
  (clean shot197)
  (empty shaker156)
  (empty shot197)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker156 l0)
  (shaker-level shaker156 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail293 ingredient73)
  (cocktail-part2 cocktail293 ingredient103)
)
 (:goal
  (and
      (contains shot197 cocktail293)
)))
