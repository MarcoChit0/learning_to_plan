(define (problem prob)
 (:domain barman)
 (:objects 
      shaker273 - shaker
      left right - hand
      shot228 shot211 - shot
      ingredient119 ingredient98 ingredient115 - ingredient
      cocktail3 - cocktail
      dispenser386 dispenser434 dispenser376 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker273)
  (ontable shot228)
  (ontable shot211)
  (dispenses dispenser386 ingredient119)
  (dispenses dispenser434 ingredient98)
  (dispenses dispenser376 ingredient115)
  (clean shaker273)
  (clean shot228)
  (clean shot211)
  (empty shaker273)
  (empty shot228)
  (empty shot211)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker273 l0)
  (shaker-level shaker273 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail3 ingredient115)
  (cocktail-part2 cocktail3 ingredient119)
)
 (:goal
  (and
      (contains shot228 cocktail3)
)))
