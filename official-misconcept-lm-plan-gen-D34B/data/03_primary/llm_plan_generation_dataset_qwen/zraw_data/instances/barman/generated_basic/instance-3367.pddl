(define (problem prob)
 (:domain barman)
 (:objects 
      shaker497 - shaker
      left right - hand
      shot380 - shot
      ingredient64 ingredient334 - ingredient
      cocktail92 - cocktail
      dispenser316 dispenser157 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker497)
  (ontable shot380)
  (dispenses dispenser316 ingredient64)
  (dispenses dispenser157 ingredient334)
  (clean shaker497)
  (clean shot380)
  (empty shaker497)
  (empty shot380)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker497 l0)
  (shaker-level shaker497 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail92 ingredient64)
  (cocktail-part2 cocktail92 ingredient334)
)
 (:goal
  (and
      (contains shot380 cocktail92)
)))
