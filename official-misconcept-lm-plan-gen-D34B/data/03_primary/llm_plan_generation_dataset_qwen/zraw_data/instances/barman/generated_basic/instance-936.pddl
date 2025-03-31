(define (problem prob)
 (:domain barman)
 (:objects 
      shaker164 - shaker
      left right - hand
      shot415 shot458 shot214 - shot
      ingredient278 ingredient208 ingredient55 ingredient427 - ingredient
      cocktail1 - cocktail
      dispenser342 dispenser317 dispenser70 dispenser160 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker164)
  (ontable shot415)
  (ontable shot458)
  (ontable shot214)
  (dispenses dispenser342 ingredient278)
  (dispenses dispenser317 ingredient208)
  (dispenses dispenser70 ingredient55)
  (dispenses dispenser160 ingredient427)
  (clean shaker164)
  (clean shot415)
  (clean shot458)
  (clean shot214)
  (empty shaker164)
  (empty shot415)
  (empty shot458)
  (empty shot214)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker164 l0)
  (shaker-level shaker164 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient278)
  (cocktail-part2 cocktail1 ingredient208)
)
 (:goal
  (and
      (contains shot415 cocktail1)
      (contains shot458 ingredient427)
)))
