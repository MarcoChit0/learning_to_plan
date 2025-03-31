(define (problem prob)
 (:domain barman)
 (:objects 
      shaker20 - shaker
      left right - hand
      shot65 shot106 shot225 - shot
      ingredient386 ingredient122 ingredient255 - ingredient
      cocktail1 - cocktail
      dispenser433 dispenser162 dispenser89 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker20)
  (ontable shot65)
  (ontable shot106)
  (ontable shot225)
  (dispenses dispenser433 ingredient386)
  (dispenses dispenser162 ingredient122)
  (dispenses dispenser89 ingredient255)
  (clean shaker20)
  (clean shot65)
  (clean shot106)
  (clean shot225)
  (empty shaker20)
  (empty shot65)
  (empty shot106)
  (empty shot225)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker20 l0)
  (shaker-level shaker20 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient255)
  (cocktail-part2 cocktail1 ingredient122)
)
 (:goal
  (and
      (contains shot65 cocktail1)
      (contains shot106 cocktail1)
)))
