(define (problem prob)
 (:domain barman)
 (:objects 
      shaker190 - shaker
      left right - hand
      shot465 - shot
      ingredient371 ingredient58 ingredient73 ingredient124 - ingredient
      cocktail441 - cocktail
      dispenser434 dispenser344 dispenser350 dispenser303 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker190)
  (ontable shot465)
  (dispenses dispenser434 ingredient371)
  (dispenses dispenser344 ingredient58)
  (dispenses dispenser350 ingredient73)
  (dispenses dispenser303 ingredient124)
  (clean shaker190)
  (clean shot465)
  (empty shaker190)
  (empty shot465)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker190 l0)
  (shaker-level shaker190 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail441 ingredient58)
  (cocktail-part2 cocktail441 ingredient371)
)
 (:goal
  (and
      (contains shot465 cocktail441)
)))
