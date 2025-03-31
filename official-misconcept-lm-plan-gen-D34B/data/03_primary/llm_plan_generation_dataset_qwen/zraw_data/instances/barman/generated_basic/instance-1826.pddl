(define (problem prob)
 (:domain barman)
 (:objects 
      shaker282 - shaker
      left right - hand
      shot311 shot10 shot22 - shot
      ingredient370 ingredient325 ingredient324 ingredient46 - ingredient
      cocktail393 - cocktail
      dispenser135 dispenser414 dispenser33 dispenser110 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker282)
  (ontable shot311)
  (ontable shot10)
  (ontable shot22)
  (dispenses dispenser135 ingredient370)
  (dispenses dispenser414 ingredient325)
  (dispenses dispenser33 ingredient324)
  (dispenses dispenser110 ingredient46)
  (clean shaker282)
  (clean shot311)
  (clean shot10)
  (clean shot22)
  (empty shaker282)
  (empty shot311)
  (empty shot10)
  (empty shot22)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker282 l0)
  (shaker-level shaker282 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail393 ingredient325)
  (cocktail-part2 cocktail393 ingredient46)
)
 (:goal
  (and
      (contains shot311 cocktail393)
      (contains shot10 ingredient46)
)))
