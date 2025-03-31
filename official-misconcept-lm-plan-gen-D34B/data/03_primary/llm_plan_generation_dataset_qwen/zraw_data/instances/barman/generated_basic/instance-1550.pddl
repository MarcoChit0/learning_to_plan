(define (problem prob)
 (:domain barman)
 (:objects 
      shaker384 - shaker
      left right - hand
      shot294 - shot
      ingredient136 ingredient195 - ingredient
      cocktail41 - cocktail
      dispenser29 dispenser422 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker384)
  (ontable shot294)
  (dispenses dispenser29 ingredient136)
  (dispenses dispenser422 ingredient195)
  (clean shaker384)
  (clean shot294)
  (empty shaker384)
  (empty shot294)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker384 l0)
  (shaker-level shaker384 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail41 ingredient195)
  (cocktail-part2 cocktail41 ingredient136)
)
 (:goal
  (and
      (contains shot294 cocktail41)
)))
