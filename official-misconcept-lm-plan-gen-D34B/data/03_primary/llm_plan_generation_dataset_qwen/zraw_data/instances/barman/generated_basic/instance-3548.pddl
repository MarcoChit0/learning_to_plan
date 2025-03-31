(define (problem prob)
 (:domain barman)
 (:objects 
      shaker219 - shaker
      left right - hand
      shot338 - shot
      ingredient401 ingredient366 ingredient427 - ingredient
      cocktail327 - cocktail
      dispenser10 dispenser3 dispenser382 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker219)
  (ontable shot338)
  (dispenses dispenser10 ingredient401)
  (dispenses dispenser3 ingredient366)
  (dispenses dispenser382 ingredient427)
  (clean shaker219)
  (clean shot338)
  (empty shaker219)
  (empty shot338)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker219 l0)
  (shaker-level shaker219 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail327 ingredient401)
  (cocktail-part2 cocktail327 ingredient427)
)
 (:goal
  (and
      (contains shot338 cocktail327)
)))
