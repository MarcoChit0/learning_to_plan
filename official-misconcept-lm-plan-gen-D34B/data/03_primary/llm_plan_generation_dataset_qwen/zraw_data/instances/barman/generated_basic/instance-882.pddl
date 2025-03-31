(define (problem prob)
 (:domain barman)
 (:objects 
      shaker132 - shaker
      left right - hand
      shot204 - shot
      ingredient50 ingredient155 - ingredient
      cocktail1 - cocktail
      dispenser195 dispenser219 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker132)
  (ontable shot204)
  (dispenses dispenser195 ingredient50)
  (dispenses dispenser219 ingredient155)
  (clean shaker132)
  (clean shot204)
  (empty shaker132)
  (empty shot204)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker132 l0)
  (shaker-level shaker132 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient50)
  (cocktail-part2 cocktail1 ingredient155)
)
 (:goal
  (and
      (contains shot204 cocktail1)
)))
