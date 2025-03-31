(define (problem prob)
 (:domain barman)
 (:objects 
      shaker429 - shaker
      left right - hand
      shot285 shot194 shot409 - shot
      ingredient293 ingredient368 ingredient275 - ingredient
      cocktail123 - cocktail
      dispenser344 dispenser386 dispenser118 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker429)
  (ontable shot285)
  (ontable shot194)
  (ontable shot409)
  (dispenses dispenser344 ingredient293)
  (dispenses dispenser386 ingredient368)
  (dispenses dispenser118 ingredient275)
  (clean shaker429)
  (clean shot285)
  (clean shot194)
  (clean shot409)
  (empty shaker429)
  (empty shot285)
  (empty shot194)
  (empty shot409)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker429 l0)
  (shaker-level shaker429 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail123 ingredient368)
  (cocktail-part2 cocktail123 ingredient293)
)
 (:goal
  (and
      (contains shot285 cocktail123)
      (contains shot194 ingredient368)
)))
