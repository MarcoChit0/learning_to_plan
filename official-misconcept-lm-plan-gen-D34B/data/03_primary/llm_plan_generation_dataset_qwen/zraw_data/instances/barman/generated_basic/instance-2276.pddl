(define (problem prob)
 (:domain barman)
 (:objects 
      shaker469 - shaker
      left right - hand
      shot273 shot479 - shot
      ingredient487 ingredient360 ingredient131 ingredient17 - ingredient
      cocktail324 - cocktail
      dispenser237 dispenser404 dispenser277 dispenser76 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker469)
  (ontable shot273)
  (ontable shot479)
  (dispenses dispenser237 ingredient487)
  (dispenses dispenser404 ingredient360)
  (dispenses dispenser277 ingredient131)
  (dispenses dispenser76 ingredient17)
  (clean shaker469)
  (clean shot273)
  (clean shot479)
  (empty shaker469)
  (empty shot273)
  (empty shot479)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker469 l0)
  (shaker-level shaker469 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail324 ingredient17)
  (cocktail-part2 cocktail324 ingredient360)
)
 (:goal
  (and
      (contains shot273 cocktail324)
)))
