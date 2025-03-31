(define (problem prob)
 (:domain barman)
 (:objects 
      shaker45 - shaker
      left right - hand
      shot137 - shot
      ingredient209 ingredient307 - ingredient
      cocktail119 - cocktail
      dispenser284 dispenser415 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker45)
  (ontable shot137)
  (dispenses dispenser284 ingredient209)
  (dispenses dispenser415 ingredient307)
  (clean shaker45)
  (clean shot137)
  (empty shaker45)
  (empty shot137)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker45 l0)
  (shaker-level shaker45 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail119 ingredient209)
  (cocktail-part2 cocktail119 ingredient307)
)
 (:goal
  (and
      (contains shot137 cocktail119)
)))
