(define (problem prob)
 (:domain barman)
 (:objects 
      shaker38 - shaker
      left right - hand
      shot94 shot490 - shot
      ingredient312 ingredient110 ingredient149 - ingredient
      cocktail456 - cocktail
      dispenser349 dispenser24 dispenser291 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker38)
  (ontable shot94)
  (ontable shot490)
  (dispenses dispenser349 ingredient312)
  (dispenses dispenser24 ingredient110)
  (dispenses dispenser291 ingredient149)
  (clean shaker38)
  (clean shot94)
  (clean shot490)
  (empty shaker38)
  (empty shot94)
  (empty shot490)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker38 l0)
  (shaker-level shaker38 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail456 ingredient110)
  (cocktail-part2 cocktail456 ingredient312)
)
 (:goal
  (and
      (contains shot94 cocktail456)
)))
