(define (problem prob)
 (:domain barman)
 (:objects 
      shaker150 - shaker
      left right - hand
      shot14 - shot
      ingredient19 ingredient204 ingredient207 - ingredient
      cocktail129 - cocktail
      dispenser220 dispenser233 dispenser111 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker150)
  (ontable shot14)
  (dispenses dispenser220 ingredient19)
  (dispenses dispenser233 ingredient204)
  (dispenses dispenser111 ingredient207)
  (clean shaker150)
  (clean shot14)
  (empty shaker150)
  (empty shot14)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker150 l0)
  (shaker-level shaker150 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail129 ingredient19)
  (cocktail-part2 cocktail129 ingredient207)
)
 (:goal
  (and
      (contains shot14 cocktail129)
)))
