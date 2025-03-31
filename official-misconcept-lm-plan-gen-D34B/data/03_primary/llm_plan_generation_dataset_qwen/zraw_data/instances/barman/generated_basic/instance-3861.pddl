(define (problem prob)
 (:domain barman)
 (:objects 
      shaker308 - shaker
      left right - hand
      shot22 shot8 - shot
      ingredient129 ingredient320 ingredient38 - ingredient
      cocktail300 - cocktail
      dispenser193 dispenser242 dispenser156 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker308)
  (ontable shot22)
  (ontable shot8)
  (dispenses dispenser193 ingredient129)
  (dispenses dispenser242 ingredient320)
  (dispenses dispenser156 ingredient38)
  (clean shaker308)
  (clean shot22)
  (clean shot8)
  (empty shaker308)
  (empty shot22)
  (empty shot8)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker308 l0)
  (shaker-level shaker308 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail300 ingredient38)
  (cocktail-part2 cocktail300 ingredient129)
)
 (:goal
  (and
      (contains shot22 cocktail300)
)))
