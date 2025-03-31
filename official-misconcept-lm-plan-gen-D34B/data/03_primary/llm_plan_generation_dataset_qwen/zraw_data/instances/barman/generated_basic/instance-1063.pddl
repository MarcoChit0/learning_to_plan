(define (problem prob)
 (:domain barman)
 (:objects 
      shaker423 - shaker
      left right - hand
      shot452 shot396 shot249 - shot
      ingredient497 ingredient408 ingredient457 ingredient462 - ingredient
      cocktail1 - cocktail
      dispenser483 dispenser44 dispenser135 dispenser426 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker423)
  (ontable shot452)
  (ontable shot396)
  (ontable shot249)
  (dispenses dispenser483 ingredient497)
  (dispenses dispenser44 ingredient408)
  (dispenses dispenser135 ingredient457)
  (dispenses dispenser426 ingredient462)
  (clean shaker423)
  (clean shot452)
  (clean shot396)
  (clean shot249)
  (empty shaker423)
  (empty shot452)
  (empty shot396)
  (empty shot249)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker423 l0)
  (shaker-level shaker423 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient497)
  (cocktail-part2 cocktail1 ingredient462)
)
 (:goal
  (and
      (contains shot452 cocktail1)
      (contains shot396 cocktail1)
)))
