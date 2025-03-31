(define (problem prob)
 (:domain barman)
 (:objects 
      shaker274 - shaker
      left right - hand
      shot55 shot193 - shot
      ingredient455 ingredient110 ingredient360 ingredient195 - ingredient
      cocktail1 - cocktail
      dispenser412 dispenser3 dispenser294 dispenser216 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker274)
  (ontable shot55)
  (ontable shot193)
  (dispenses dispenser412 ingredient455)
  (dispenses dispenser3 ingredient110)
  (dispenses dispenser294 ingredient360)
  (dispenses dispenser216 ingredient195)
  (clean shaker274)
  (clean shot55)
  (clean shot193)
  (empty shaker274)
  (empty shot55)
  (empty shot193)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker274 l0)
  (shaker-level shaker274 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient195)
  (cocktail-part2 cocktail1 ingredient360)
)
 (:goal
  (and
      (contains shot55 cocktail1)
)))
