(define (problem prob)
 (:domain barman)
 (:objects 
      shaker480 - shaker
      left right - hand
      shot127 - shot
      ingredient444 ingredient312 ingredient487 ingredient228 - ingredient
      cocktail48 - cocktail
      dispenser110 dispenser111 dispenser248 dispenser478 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker480)
  (ontable shot127)
  (dispenses dispenser110 ingredient444)
  (dispenses dispenser111 ingredient312)
  (dispenses dispenser248 ingredient487)
  (dispenses dispenser478 ingredient228)
  (clean shaker480)
  (clean shot127)
  (empty shaker480)
  (empty shot127)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker480 l0)
  (shaker-level shaker480 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail48 ingredient228)
  (cocktail-part2 cocktail48 ingredient487)
)
 (:goal
  (and
      (contains shot127 cocktail48)
)))
