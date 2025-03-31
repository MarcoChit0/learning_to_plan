(define (problem prob)
 (:domain barman)
 (:objects 
      shaker466 - shaker
      left right - hand
      shot98 shot323 shot99 - shot
      ingredient106 ingredient37 - ingredient
      cocktail1 - cocktail
      dispenser372 dispenser151 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker466)
  (ontable shot98)
  (ontable shot323)
  (ontable shot99)
  (dispenses dispenser372 ingredient106)
  (dispenses dispenser151 ingredient37)
  (clean shaker466)
  (clean shot98)
  (clean shot323)
  (clean shot99)
  (empty shaker466)
  (empty shot98)
  (empty shot323)
  (empty shot99)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker466 l0)
  (shaker-level shaker466 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient37)
  (cocktail-part2 cocktail1 ingredient106)
)
 (:goal
  (and
      (contains shot98 cocktail1)
      (contains shot323 ingredient37)
)))
