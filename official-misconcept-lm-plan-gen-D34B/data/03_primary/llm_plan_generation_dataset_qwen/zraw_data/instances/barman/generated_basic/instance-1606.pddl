(define (problem prob)
 (:domain barman)
 (:objects 
      shaker210 - shaker
      left right - hand
      shot194 shot57 - shot
      ingredient6 ingredient260 ingredient89 - ingredient
      cocktail465 - cocktail
      dispenser216 dispenser102 dispenser243 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker210)
  (ontable shot194)
  (ontable shot57)
  (dispenses dispenser216 ingredient6)
  (dispenses dispenser102 ingredient260)
  (dispenses dispenser243 ingredient89)
  (clean shaker210)
  (clean shot194)
  (clean shot57)
  (empty shaker210)
  (empty shot194)
  (empty shot57)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker210 l0)
  (shaker-level shaker210 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail465 ingredient260)
  (cocktail-part2 cocktail465 ingredient89)
)
 (:goal
  (and
      (contains shot194 cocktail465)
)))
