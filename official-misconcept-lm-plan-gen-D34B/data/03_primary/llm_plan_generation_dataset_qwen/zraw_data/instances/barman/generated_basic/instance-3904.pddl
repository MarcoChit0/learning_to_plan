(define (problem prob)
 (:domain barman)
 (:objects 
      shaker238 - shaker
      left right - hand
      shot75 - shot
      ingredient315 ingredient162 ingredient401 - ingredient
      cocktail454 - cocktail
      dispenser367 dispenser101 dispenser409 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker238)
  (ontable shot75)
  (dispenses dispenser367 ingredient315)
  (dispenses dispenser101 ingredient162)
  (dispenses dispenser409 ingredient401)
  (clean shaker238)
  (clean shot75)
  (empty shaker238)
  (empty shot75)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker238 l0)
  (shaker-level shaker238 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail454 ingredient401)
  (cocktail-part2 cocktail454 ingredient162)
)
 (:goal
  (and
      (contains shot75 cocktail454)
)))
