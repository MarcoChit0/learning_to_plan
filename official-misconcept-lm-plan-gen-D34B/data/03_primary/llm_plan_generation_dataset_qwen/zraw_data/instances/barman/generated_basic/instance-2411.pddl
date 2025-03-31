(define (problem prob)
 (:domain barman)
 (:objects 
      shaker263 - shaker
      left right - hand
      shot294 - shot
      ingredient10 ingredient67 - ingredient
      cocktail69 - cocktail
      dispenser194 dispenser364 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker263)
  (ontable shot294)
  (dispenses dispenser194 ingredient10)
  (dispenses dispenser364 ingredient67)
  (clean shaker263)
  (clean shot294)
  (empty shaker263)
  (empty shot294)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker263 l0)
  (shaker-level shaker263 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail69 ingredient67)
  (cocktail-part2 cocktail69 ingredient10)
)
 (:goal
  (and
      (contains shot294 cocktail69)
)))
