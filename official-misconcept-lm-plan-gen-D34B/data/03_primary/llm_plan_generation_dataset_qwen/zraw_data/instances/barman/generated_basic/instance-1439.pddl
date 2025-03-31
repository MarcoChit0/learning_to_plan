(define (problem prob)
 (:domain barman)
 (:objects 
      shaker243 - shaker
      left right - hand
      shot27 - shot
      ingredient404 ingredient192 ingredient402 - ingredient
      cocktail400 - cocktail
      dispenser476 dispenser44 dispenser148 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker243)
  (ontable shot27)
  (dispenses dispenser476 ingredient404)
  (dispenses dispenser44 ingredient192)
  (dispenses dispenser148 ingredient402)
  (clean shaker243)
  (clean shot27)
  (empty shaker243)
  (empty shot27)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker243 l0)
  (shaker-level shaker243 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail400 ingredient404)
  (cocktail-part2 cocktail400 ingredient402)
)
 (:goal
  (and
      (contains shot27 cocktail400)
)))
