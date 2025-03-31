(define (problem prob)
 (:domain barman)
 (:objects 
      shaker256 - shaker
      left right - hand
      shot280 shot490 - shot
      ingredient149 ingredient429 ingredient230 - ingredient
      cocktail1 - cocktail
      dispenser238 dispenser480 dispenser109 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker256)
  (ontable shot280)
  (ontable shot490)
  (dispenses dispenser238 ingredient149)
  (dispenses dispenser480 ingredient429)
  (dispenses dispenser109 ingredient230)
  (clean shaker256)
  (clean shot280)
  (clean shot490)
  (empty shaker256)
  (empty shot280)
  (empty shot490)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker256 l0)
  (shaker-level shaker256 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient429)
  (cocktail-part2 cocktail1 ingredient230)
)
 (:goal
  (and
      (contains shot280 cocktail1)
)))
