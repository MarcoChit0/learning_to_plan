(define (problem prob)
 (:domain barman)
 (:objects 
      shaker207 - shaker
      left right - hand
      shot125 shot136 - shot
      ingredient471 ingredient54 ingredient383 - ingredient
      cocktail382 - cocktail
      dispenser448 dispenser43 dispenser215 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker207)
  (ontable shot125)
  (ontable shot136)
  (dispenses dispenser448 ingredient471)
  (dispenses dispenser43 ingredient54)
  (dispenses dispenser215 ingredient383)
  (clean shaker207)
  (clean shot125)
  (clean shot136)
  (empty shaker207)
  (empty shot125)
  (empty shot136)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker207 l0)
  (shaker-level shaker207 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail382 ingredient54)
  (cocktail-part2 cocktail382 ingredient383)
)
 (:goal
  (and
      (contains shot125 cocktail382)
)))
