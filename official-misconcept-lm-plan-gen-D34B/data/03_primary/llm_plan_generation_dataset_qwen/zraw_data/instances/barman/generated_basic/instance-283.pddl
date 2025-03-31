(define (problem prob)
 (:domain barman)
 (:objects 
      shaker295 - shaker
      left right - hand
      shot360 shot476 - shot
      ingredient238 ingredient25 ingredient31 - ingredient
      cocktail1 - cocktail
      dispenser478 dispenser167 dispenser70 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker295)
  (ontable shot360)
  (ontable shot476)
  (dispenses dispenser478 ingredient238)
  (dispenses dispenser167 ingredient25)
  (dispenses dispenser70 ingredient31)
  (clean shaker295)
  (clean shot360)
  (clean shot476)
  (empty shaker295)
  (empty shot360)
  (empty shot476)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker295 l0)
  (shaker-level shaker295 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient238)
  (cocktail-part2 cocktail1 ingredient25)
)
 (:goal
  (and
      (contains shot360 cocktail1)
)))
