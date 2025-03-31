(define (problem prob)
 (:domain barman)
 (:objects 
      shaker179 - shaker
      left right - hand
      shot37 shot66 - shot
      ingredient205 ingredient28 ingredient80 - ingredient
      cocktail1 - cocktail
      dispenser216 dispenser272 dispenser341 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker179)
  (ontable shot37)
  (ontable shot66)
  (dispenses dispenser216 ingredient205)
  (dispenses dispenser272 ingredient28)
  (dispenses dispenser341 ingredient80)
  (clean shaker179)
  (clean shot37)
  (clean shot66)
  (empty shaker179)
  (empty shot37)
  (empty shot66)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker179 l0)
  (shaker-level shaker179 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient80)
  (cocktail-part2 cocktail1 ingredient28)
)
 (:goal
  (and
      (contains shot37 cocktail1)
)))
