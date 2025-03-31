(define (problem prob)
 (:domain barman)
 (:objects 
      shaker281 - shaker
      left right - hand
      shot42 shot254 shot135 - shot
      ingredient55 ingredient17 ingredient93 ingredient470 - ingredient
      cocktail88 - cocktail
      dispenser99 dispenser0 dispenser360 dispenser152 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker281)
  (ontable shot42)
  (ontable shot254)
  (ontable shot135)
  (dispenses dispenser99 ingredient55)
  (dispenses dispenser0 ingredient17)
  (dispenses dispenser360 ingredient93)
  (dispenses dispenser152 ingredient470)
  (clean shaker281)
  (clean shot42)
  (clean shot254)
  (clean shot135)
  (empty shaker281)
  (empty shot42)
  (empty shot254)
  (empty shot135)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker281 l0)
  (shaker-level shaker281 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail88 ingredient470)
  (cocktail-part2 cocktail88 ingredient17)
)
 (:goal
  (and
      (contains shot42 cocktail88)
      (contains shot254 ingredient55)
)))
