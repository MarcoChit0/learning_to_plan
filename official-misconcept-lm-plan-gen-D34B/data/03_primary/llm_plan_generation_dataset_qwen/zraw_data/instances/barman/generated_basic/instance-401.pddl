(define (problem prob)
 (:domain barman)
 (:objects 
      shaker56 - shaker
      left right - hand
      shot222 shot62 - shot
      ingredient167 ingredient50 - ingredient
      cocktail1 - cocktail
      dispenser90 dispenser341 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker56)
  (ontable shot222)
  (ontable shot62)
  (dispenses dispenser90 ingredient167)
  (dispenses dispenser341 ingredient50)
  (clean shaker56)
  (clean shot222)
  (clean shot62)
  (empty shaker56)
  (empty shot222)
  (empty shot62)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker56 l0)
  (shaker-level shaker56 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient50)
  (cocktail-part2 cocktail1 ingredient167)
)
 (:goal
  (and
      (contains shot222 cocktail1)
)))
