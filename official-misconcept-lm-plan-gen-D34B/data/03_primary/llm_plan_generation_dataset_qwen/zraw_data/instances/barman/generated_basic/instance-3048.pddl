(define (problem prob)
 (:domain barman)
 (:objects 
      shaker385 - shaker
      left right - hand
      shot433 shot125 - shot
      ingredient220 ingredient127 ingredient101 - ingredient
      cocktail116 - cocktail
      dispenser124 dispenser483 dispenser291 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker385)
  (ontable shot433)
  (ontable shot125)
  (dispenses dispenser124 ingredient220)
  (dispenses dispenser483 ingredient127)
  (dispenses dispenser291 ingredient101)
  (clean shaker385)
  (clean shot433)
  (clean shot125)
  (empty shaker385)
  (empty shot433)
  (empty shot125)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker385 l0)
  (shaker-level shaker385 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail116 ingredient127)
  (cocktail-part2 cocktail116 ingredient220)
)
 (:goal
  (and
      (contains shot433 cocktail116)
)))
