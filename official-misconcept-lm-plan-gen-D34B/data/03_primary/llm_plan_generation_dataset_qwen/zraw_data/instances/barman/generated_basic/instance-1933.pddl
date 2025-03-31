(define (problem prob)
 (:domain barman)
 (:objects 
      shaker328 - shaker
      left right - hand
      shot452 - shot
      ingredient253 ingredient327 - ingredient
      cocktail109 - cocktail
      dispenser212 dispenser78 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker328)
  (ontable shot452)
  (dispenses dispenser212 ingredient253)
  (dispenses dispenser78 ingredient327)
  (clean shaker328)
  (clean shot452)
  (empty shaker328)
  (empty shot452)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker328 l0)
  (shaker-level shaker328 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail109 ingredient327)
  (cocktail-part2 cocktail109 ingredient253)
)
 (:goal
  (and
      (contains shot452 cocktail109)
)))
