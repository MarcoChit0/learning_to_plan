(define (problem prob)
 (:domain barman)
 (:objects 
      shaker461 - shaker
      left right - hand
      shot425 - shot
      ingredient492 ingredient9 - ingredient
      cocktail469 - cocktail
      dispenser272 dispenser237 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker461)
  (ontable shot425)
  (dispenses dispenser272 ingredient492)
  (dispenses dispenser237 ingredient9)
  (clean shaker461)
  (clean shot425)
  (empty shaker461)
  (empty shot425)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker461 l0)
  (shaker-level shaker461 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail469 ingredient492)
  (cocktail-part2 cocktail469 ingredient9)
)
 (:goal
  (and
      (contains shot425 cocktail469)
)))
