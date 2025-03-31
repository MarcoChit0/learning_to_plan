(define (problem prob)
 (:domain barman)
 (:objects 
      shaker40 - shaker
      left right - hand
      shot186 - shot
      ingredient254 ingredient173 ingredient334 ingredient299 - ingredient
      cocktail1 - cocktail
      dispenser103 dispenser19 dispenser123 dispenser5 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker40)
  (ontable shot186)
  (dispenses dispenser103 ingredient254)
  (dispenses dispenser19 ingredient173)
  (dispenses dispenser123 ingredient334)
  (dispenses dispenser5 ingredient299)
  (clean shaker40)
  (clean shot186)
  (empty shaker40)
  (empty shot186)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker40 l0)
  (shaker-level shaker40 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient173)
  (cocktail-part2 cocktail1 ingredient254)
)
 (:goal
  (and
      (contains shot186 cocktail1)
)))
