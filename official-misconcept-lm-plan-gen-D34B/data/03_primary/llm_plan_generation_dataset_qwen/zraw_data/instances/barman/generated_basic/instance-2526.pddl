(define (problem prob)
 (:domain barman)
 (:objects 
      shaker288 - shaker
      left right - hand
      shot253 shot120 shot8 - shot
      ingredient415 ingredient80 - ingredient
      cocktail5 - cocktail
      dispenser318 dispenser465 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker288)
  (ontable shot253)
  (ontable shot120)
  (ontable shot8)
  (dispenses dispenser318 ingredient415)
  (dispenses dispenser465 ingredient80)
  (clean shaker288)
  (clean shot253)
  (clean shot120)
  (clean shot8)
  (empty shaker288)
  (empty shot253)
  (empty shot120)
  (empty shot8)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker288 l0)
  (shaker-level shaker288 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail5 ingredient415)
  (cocktail-part2 cocktail5 ingredient80)
)
 (:goal
  (and
      (contains shot253 cocktail5)
      (contains shot120 ingredient415)
)))
