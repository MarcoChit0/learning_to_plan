(define (problem prob)
 (:domain barman)
 (:objects 
      shaker81 - shaker
      left right - hand
      shot201 shot194 shot457 - shot
      ingredient229 ingredient155 ingredient102 ingredient58 - ingredient
      cocktail1 - cocktail
      dispenser279 dispenser406 dispenser245 dispenser96 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker81)
  (ontable shot201)
  (ontable shot194)
  (ontable shot457)
  (dispenses dispenser279 ingredient229)
  (dispenses dispenser406 ingredient155)
  (dispenses dispenser245 ingredient102)
  (dispenses dispenser96 ingredient58)
  (clean shaker81)
  (clean shot201)
  (clean shot194)
  (clean shot457)
  (empty shaker81)
  (empty shot201)
  (empty shot194)
  (empty shot457)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker81 l0)
  (shaker-level shaker81 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient58)
  (cocktail-part2 cocktail1 ingredient155)
)
 (:goal
  (and
      (contains shot201 cocktail1)
      (contains shot194 ingredient155)
)))
