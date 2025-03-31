(define (problem prob)
 (:domain barman)
 (:objects 
      shaker472 - shaker
      left right - hand
      shot431 shot299 - shot
      ingredient80 ingredient331 ingredient482 - ingredient
      cocktail330 - cocktail
      dispenser250 dispenser261 dispenser194 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker472)
  (ontable shot431)
  (ontable shot299)
  (dispenses dispenser250 ingredient80)
  (dispenses dispenser261 ingredient331)
  (dispenses dispenser194 ingredient482)
  (clean shaker472)
  (clean shot431)
  (clean shot299)
  (empty shaker472)
  (empty shot431)
  (empty shot299)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker472 l0)
  (shaker-level shaker472 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail330 ingredient331)
  (cocktail-part2 cocktail330 ingredient80)
)
 (:goal
  (and
      (contains shot431 cocktail330)
)))
