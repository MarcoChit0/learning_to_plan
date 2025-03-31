(define (problem prob)
 (:domain barman)
 (:objects 
      shaker26 - shaker
      left right - hand
      shot170 shot204 - shot
      ingredient406 ingredient24 - ingredient
      cocktail366 - cocktail
      dispenser219 dispenser59 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker26)
  (ontable shot170)
  (ontable shot204)
  (dispenses dispenser219 ingredient406)
  (dispenses dispenser59 ingredient24)
  (clean shaker26)
  (clean shot170)
  (clean shot204)
  (empty shaker26)
  (empty shot170)
  (empty shot204)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker26 l0)
  (shaker-level shaker26 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail366 ingredient24)
  (cocktail-part2 cocktail366 ingredient406)
)
 (:goal
  (and
      (contains shot170 cocktail366)
)))
