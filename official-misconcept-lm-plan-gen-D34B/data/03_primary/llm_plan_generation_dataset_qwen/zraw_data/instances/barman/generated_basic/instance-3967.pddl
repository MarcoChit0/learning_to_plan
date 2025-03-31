(define (problem prob)
 (:domain barman)
 (:objects 
      shaker490 - shaker
      left right - hand
      shot388 shot43 - shot
      ingredient101 ingredient39 ingredient428 ingredient149 - ingredient
      cocktail37 - cocktail
      dispenser52 dispenser203 dispenser275 dispenser367 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker490)
  (ontable shot388)
  (ontable shot43)
  (dispenses dispenser52 ingredient101)
  (dispenses dispenser203 ingredient39)
  (dispenses dispenser275 ingredient428)
  (dispenses dispenser367 ingredient149)
  (clean shaker490)
  (clean shot388)
  (clean shot43)
  (empty shaker490)
  (empty shot388)
  (empty shot43)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker490 l0)
  (shaker-level shaker490 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail37 ingredient149)
  (cocktail-part2 cocktail37 ingredient428)
)
 (:goal
  (and
      (contains shot388 cocktail37)
)))
