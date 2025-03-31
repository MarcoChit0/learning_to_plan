(define (problem prob)
 (:domain barman)
 (:objects 
      shaker211 - shaker
      left right - hand
      shot442 shot147 - shot
      ingredient131 ingredient283 ingredient355 - ingredient
      cocktail1 - cocktail
      dispenser42 dispenser193 dispenser81 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker211)
  (ontable shot442)
  (ontable shot147)
  (dispenses dispenser42 ingredient131)
  (dispenses dispenser193 ingredient283)
  (dispenses dispenser81 ingredient355)
  (clean shaker211)
  (clean shot442)
  (clean shot147)
  (empty shaker211)
  (empty shot442)
  (empty shot147)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker211 l0)
  (shaker-level shaker211 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient283)
  (cocktail-part2 cocktail1 ingredient131)
)
 (:goal
  (and
      (contains shot442 cocktail1)
)))
