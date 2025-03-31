(define (problem prob)
 (:domain barman)
 (:objects 
      shaker51 - shaker
      left right - hand
      shot394 - shot
      ingredient272 ingredient423 - ingredient
      cocktail1 - cocktail
      dispenser394 dispenser398 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker51)
  (ontable shot394)
  (dispenses dispenser394 ingredient272)
  (dispenses dispenser398 ingredient423)
  (clean shaker51)
  (clean shot394)
  (empty shaker51)
  (empty shot394)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker51 l0)
  (shaker-level shaker51 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient272)
  (cocktail-part2 cocktail1 ingredient423)
)
 (:goal
  (and
      (contains shot394 cocktail1)
)))
