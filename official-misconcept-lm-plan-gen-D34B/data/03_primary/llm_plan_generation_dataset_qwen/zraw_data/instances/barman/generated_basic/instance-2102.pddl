(define (problem prob)
 (:domain barman)
 (:objects 
      shaker219 - shaker
      left right - hand
      shot296 shot366 shot132 - shot
      ingredient53 ingredient480 - ingredient
      cocktail391 - cocktail
      dispenser322 dispenser131 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker219)
  (ontable shot296)
  (ontable shot366)
  (ontable shot132)
  (dispenses dispenser322 ingredient53)
  (dispenses dispenser131 ingredient480)
  (clean shaker219)
  (clean shot296)
  (clean shot366)
  (clean shot132)
  (empty shaker219)
  (empty shot296)
  (empty shot366)
  (empty shot132)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker219 l0)
  (shaker-level shaker219 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail391 ingredient480)
  (cocktail-part2 cocktail391 ingredient53)
)
 (:goal
  (and
      (contains shot296 cocktail391)
      (contains shot366 cocktail391)
)))
