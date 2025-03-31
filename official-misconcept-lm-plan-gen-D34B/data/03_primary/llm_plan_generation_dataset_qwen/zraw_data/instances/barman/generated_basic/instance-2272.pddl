(define (problem prob)
 (:domain barman)
 (:objects 
      shaker294 - shaker
      left right - hand
      shot339 shot131 - shot
      ingredient374 ingredient285 ingredient479 ingredient347 - ingredient
      cocktail356 - cocktail
      dispenser197 dispenser407 dispenser152 dispenser265 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker294)
  (ontable shot339)
  (ontable shot131)
  (dispenses dispenser197 ingredient374)
  (dispenses dispenser407 ingredient285)
  (dispenses dispenser152 ingredient479)
  (dispenses dispenser265 ingredient347)
  (clean shaker294)
  (clean shot339)
  (clean shot131)
  (empty shaker294)
  (empty shot339)
  (empty shot131)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker294 l0)
  (shaker-level shaker294 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail356 ingredient374)
  (cocktail-part2 cocktail356 ingredient285)
)
 (:goal
  (and
      (contains shot339 cocktail356)
)))
