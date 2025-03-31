(define (problem prob)
 (:domain barman)
 (:objects 
      shaker330 - shaker
      left right - hand
      shot457 shot189 - shot
      ingredient245 ingredient427 ingredient184 - ingredient
      cocktail231 - cocktail
      dispenser426 dispenser454 dispenser125 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker330)
  (ontable shot457)
  (ontable shot189)
  (dispenses dispenser426 ingredient245)
  (dispenses dispenser454 ingredient427)
  (dispenses dispenser125 ingredient184)
  (clean shaker330)
  (clean shot457)
  (clean shot189)
  (empty shaker330)
  (empty shot457)
  (empty shot189)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker330 l0)
  (shaker-level shaker330 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail231 ingredient184)
  (cocktail-part2 cocktail231 ingredient427)
)
 (:goal
  (and
      (contains shot457 cocktail231)
)))
