(define (problem prob)
 (:domain barman)
 (:objects 
      shaker107 - shaker
      left right - hand
      shot297 shot5 shot442 - shot
      ingredient468 ingredient324 ingredient73 ingredient80 - ingredient
      cocktail241 - cocktail
      dispenser34 dispenser124 dispenser262 dispenser45 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker107)
  (ontable shot297)
  (ontable shot5)
  (ontable shot442)
  (dispenses dispenser34 ingredient468)
  (dispenses dispenser124 ingredient324)
  (dispenses dispenser262 ingredient73)
  (dispenses dispenser45 ingredient80)
  (clean shaker107)
  (clean shot297)
  (clean shot5)
  (clean shot442)
  (empty shaker107)
  (empty shot297)
  (empty shot5)
  (empty shot442)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker107 l0)
  (shaker-level shaker107 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail241 ingredient324)
  (cocktail-part2 cocktail241 ingredient73)
)
 (:goal
  (and
      (contains shot297 cocktail241)
      (contains shot5 cocktail241)
)))
