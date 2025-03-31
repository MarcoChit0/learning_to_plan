(define (problem prob)
 (:domain barman)
 (:objects 
      shaker152 - shaker
      left right - hand
      shot276 shot363 shot204 - shot
      ingredient142 ingredient4 ingredient387 ingredient8 - ingredient
      cocktail108 - cocktail
      dispenser385 dispenser28 dispenser317 dispenser480 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker152)
  (ontable shot276)
  (ontable shot363)
  (ontable shot204)
  (dispenses dispenser385 ingredient142)
  (dispenses dispenser28 ingredient4)
  (dispenses dispenser317 ingredient387)
  (dispenses dispenser480 ingredient8)
  (clean shaker152)
  (clean shot276)
  (clean shot363)
  (clean shot204)
  (empty shaker152)
  (empty shot276)
  (empty shot363)
  (empty shot204)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker152 l0)
  (shaker-level shaker152 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail108 ingredient387)
  (cocktail-part2 cocktail108 ingredient4)
)
 (:goal
  (and
      (contains shot276 cocktail108)
      (contains shot363 cocktail108)
)))
