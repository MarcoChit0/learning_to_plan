(define (problem prob)
 (:domain barman)
 (:objects 
      shaker256 - shaker
      left right - hand
      shot431 - shot
      ingredient81 ingredient399 - ingredient
      cocktail79 - cocktail
      dispenser414 dispenser417 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker256)
  (ontable shot431)
  (dispenses dispenser414 ingredient81)
  (dispenses dispenser417 ingredient399)
  (clean shaker256)
  (clean shot431)
  (empty shaker256)
  (empty shot431)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker256 l0)
  (shaker-level shaker256 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail79 ingredient81)
  (cocktail-part2 cocktail79 ingredient399)
)
 (:goal
  (and
      (contains shot431 cocktail79)
)))
