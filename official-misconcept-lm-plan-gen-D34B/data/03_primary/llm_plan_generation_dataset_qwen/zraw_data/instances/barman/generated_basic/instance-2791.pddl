(define (problem prob)
 (:domain barman)
 (:objects 
      shaker137 - shaker
      left right - hand
      shot26 - shot
      ingredient54 ingredient137 ingredient481 ingredient140 - ingredient
      cocktail85 - cocktail
      dispenser382 dispenser101 dispenser171 dispenser434 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker137)
  (ontable shot26)
  (dispenses dispenser382 ingredient54)
  (dispenses dispenser101 ingredient137)
  (dispenses dispenser171 ingredient481)
  (dispenses dispenser434 ingredient140)
  (clean shaker137)
  (clean shot26)
  (empty shaker137)
  (empty shot26)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker137 l0)
  (shaker-level shaker137 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail85 ingredient140)
  (cocktail-part2 cocktail85 ingredient54)
)
 (:goal
  (and
      (contains shot26 cocktail85)
)))
