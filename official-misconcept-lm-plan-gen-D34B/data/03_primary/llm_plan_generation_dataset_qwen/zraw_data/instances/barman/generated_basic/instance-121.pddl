(define (problem prob)
 (:domain barman)
 (:objects 
      shaker445 - shaker
      left right - hand
      shot17 shot382 - shot
      ingredient118 ingredient466 - ingredient
      cocktail1 - cocktail
      dispenser271 dispenser168 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker445)
  (ontable shot17)
  (ontable shot382)
  (dispenses dispenser271 ingredient118)
  (dispenses dispenser168 ingredient466)
  (clean shaker445)
  (clean shot17)
  (clean shot382)
  (empty shaker445)
  (empty shot17)
  (empty shot382)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker445 l0)
  (shaker-level shaker445 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient118)
  (cocktail-part2 cocktail1 ingredient466)
)
 (:goal
  (and
      (contains shot17 cocktail1)
)))
