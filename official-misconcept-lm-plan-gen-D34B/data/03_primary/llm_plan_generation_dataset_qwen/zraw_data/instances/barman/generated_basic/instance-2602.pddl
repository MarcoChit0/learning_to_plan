(define (problem prob)
 (:domain barman)
 (:objects 
      shaker250 - shaker
      left right - hand
      shot258 shot242 - shot
      ingredient348 ingredient4 ingredient430 - ingredient
      cocktail86 - cocktail
      dispenser120 dispenser278 dispenser293 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker250)
  (ontable shot258)
  (ontable shot242)
  (dispenses dispenser120 ingredient348)
  (dispenses dispenser278 ingredient4)
  (dispenses dispenser293 ingredient430)
  (clean shaker250)
  (clean shot258)
  (clean shot242)
  (empty shaker250)
  (empty shot258)
  (empty shot242)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker250 l0)
  (shaker-level shaker250 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail86 ingredient348)
  (cocktail-part2 cocktail86 ingredient430)
)
 (:goal
  (and
      (contains shot258 cocktail86)
)))
