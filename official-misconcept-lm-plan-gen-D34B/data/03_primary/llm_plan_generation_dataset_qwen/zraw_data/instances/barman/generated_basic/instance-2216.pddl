(define (problem prob)
 (:domain barman)
 (:objects 
      shaker275 - shaker
      left right - hand
      shot398 shot387 shot165 - shot
      ingredient478 ingredient308 ingredient28 ingredient101 - ingredient
      cocktail195 - cocktail
      dispenser182 dispenser58 dispenser82 dispenser205 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker275)
  (ontable shot398)
  (ontable shot387)
  (ontable shot165)
  (dispenses dispenser182 ingredient478)
  (dispenses dispenser58 ingredient308)
  (dispenses dispenser82 ingredient28)
  (dispenses dispenser205 ingredient101)
  (clean shaker275)
  (clean shot398)
  (clean shot387)
  (clean shot165)
  (empty shaker275)
  (empty shot398)
  (empty shot387)
  (empty shot165)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker275 l0)
  (shaker-level shaker275 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail195 ingredient308)
  (cocktail-part2 cocktail195 ingredient478)
)
 (:goal
  (and
      (contains shot398 cocktail195)
      (contains shot387 ingredient101)
)))
