(define (problem prob)
 (:domain barman)
 (:objects 
      shaker308 - shaker
      left right - hand
      shot222 - shot
      ingredient135 ingredient191 - ingredient
      cocktail376 - cocktail
      dispenser86 dispenser478 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker308)
  (ontable shot222)
  (dispenses dispenser86 ingredient135)
  (dispenses dispenser478 ingredient191)
  (clean shaker308)
  (clean shot222)
  (empty shaker308)
  (empty shot222)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker308 l0)
  (shaker-level shaker308 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail376 ingredient135)
  (cocktail-part2 cocktail376 ingredient191)
)
 (:goal
  (and
      (contains shot222 cocktail376)
)))
