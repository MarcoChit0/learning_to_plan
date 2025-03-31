(define (problem prob)
 (:domain barman)
 (:objects 
      shaker76 - shaker
      left right - hand
      shot328 shot201 - shot
      ingredient396 ingredient482 - ingredient
      cocktail217 - cocktail
      dispenser347 dispenser465 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker76)
  (ontable shot328)
  (ontable shot201)
  (dispenses dispenser347 ingredient396)
  (dispenses dispenser465 ingredient482)
  (clean shaker76)
  (clean shot328)
  (clean shot201)
  (empty shaker76)
  (empty shot328)
  (empty shot201)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker76 l0)
  (shaker-level shaker76 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail217 ingredient482)
  (cocktail-part2 cocktail217 ingredient396)
)
 (:goal
  (and
      (contains shot328 cocktail217)
)))
