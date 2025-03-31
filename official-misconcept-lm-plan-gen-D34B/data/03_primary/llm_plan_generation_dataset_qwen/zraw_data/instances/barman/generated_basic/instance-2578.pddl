(define (problem prob)
 (:domain barman)
 (:objects 
      shaker382 - shaker
      left right - hand
      shot344 - shot
      ingredient231 ingredient59 - ingredient
      cocktail367 - cocktail
      dispenser152 dispenser458 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker382)
  (ontable shot344)
  (dispenses dispenser152 ingredient231)
  (dispenses dispenser458 ingredient59)
  (clean shaker382)
  (clean shot344)
  (empty shaker382)
  (empty shot344)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker382 l0)
  (shaker-level shaker382 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail367 ingredient59)
  (cocktail-part2 cocktail367 ingredient231)
)
 (:goal
  (and
      (contains shot344 cocktail367)
)))
