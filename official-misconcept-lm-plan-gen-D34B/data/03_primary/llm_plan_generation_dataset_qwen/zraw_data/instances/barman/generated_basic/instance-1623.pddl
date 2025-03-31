(define (problem prob)
 (:domain barman)
 (:objects 
      shaker438 - shaker
      left right - hand
      shot491 shot26 shot16 - shot
      ingredient190 ingredient267 - ingredient
      cocktail159 - cocktail
      dispenser462 dispenser151 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker438)
  (ontable shot491)
  (ontable shot26)
  (ontable shot16)
  (dispenses dispenser462 ingredient190)
  (dispenses dispenser151 ingredient267)
  (clean shaker438)
  (clean shot491)
  (clean shot26)
  (clean shot16)
  (empty shaker438)
  (empty shot491)
  (empty shot26)
  (empty shot16)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker438 l0)
  (shaker-level shaker438 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail159 ingredient190)
  (cocktail-part2 cocktail159 ingredient267)
)
 (:goal
  (and
      (contains shot491 cocktail159)
      (contains shot26 cocktail159)
)))
