(define (problem prob)
 (:domain barman)
 (:objects 
      shaker172 - shaker
      left right - hand
      shot344 shot138 shot273 - shot
      ingredient166 ingredient193 ingredient35 ingredient395 - ingredient
      cocktail119 - cocktail
      dispenser90 dispenser123 dispenser490 dispenser240 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker172)
  (ontable shot344)
  (ontable shot138)
  (ontable shot273)
  (dispenses dispenser90 ingredient166)
  (dispenses dispenser123 ingredient193)
  (dispenses dispenser490 ingredient35)
  (dispenses dispenser240 ingredient395)
  (clean shaker172)
  (clean shot344)
  (clean shot138)
  (clean shot273)
  (empty shaker172)
  (empty shot344)
  (empty shot138)
  (empty shot273)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker172 l0)
  (shaker-level shaker172 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail119 ingredient193)
  (cocktail-part2 cocktail119 ingredient35)
)
 (:goal
  (and
      (contains shot344 cocktail119)
      (contains shot138 ingredient166)
)))
