(define (problem prob)
 (:domain barman)
 (:objects 
      shaker370 - shaker
      left right - hand
      shot275 shot474 - shot
      ingredient342 ingredient126 ingredient442 - ingredient
      cocktail1 - cocktail
      dispenser73 dispenser92 dispenser418 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker370)
  (ontable shot275)
  (ontable shot474)
  (dispenses dispenser73 ingredient342)
  (dispenses dispenser92 ingredient126)
  (dispenses dispenser418 ingredient442)
  (clean shaker370)
  (clean shot275)
  (clean shot474)
  (empty shaker370)
  (empty shot275)
  (empty shot474)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker370 l0)
  (shaker-level shaker370 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient342)
  (cocktail-part2 cocktail1 ingredient126)
)
 (:goal
  (and
      (contains shot275 cocktail1)
)))
