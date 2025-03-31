(define (problem prob)
 (:domain barman)
 (:objects 
      shaker75 - shaker
      left right - hand
      shot163 - shot
      ingredient62 ingredient198 - ingredient
      cocktail2 - cocktail
      dispenser253 dispenser338 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker75)
  (ontable shot163)
  (dispenses dispenser253 ingredient62)
  (dispenses dispenser338 ingredient198)
  (clean shaker75)
  (clean shot163)
  (empty shaker75)
  (empty shot163)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker75 l0)
  (shaker-level shaker75 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail2 ingredient62)
  (cocktail-part2 cocktail2 ingredient198)
)
 (:goal
  (and
      (contains shot163 cocktail2)
)))
