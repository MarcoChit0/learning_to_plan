(define (problem prob)
 (:domain barman)
 (:objects 
      shaker150 - shaker
      left right - hand
      shot470 shot238 - shot
      ingredient306 ingredient177 ingredient248 ingredient381 - ingredient
      cocktail263 - cocktail
      dispenser337 dispenser378 dispenser48 dispenser296 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker150)
  (ontable shot470)
  (ontable shot238)
  (dispenses dispenser337 ingredient306)
  (dispenses dispenser378 ingredient177)
  (dispenses dispenser48 ingredient248)
  (dispenses dispenser296 ingredient381)
  (clean shaker150)
  (clean shot470)
  (clean shot238)
  (empty shaker150)
  (empty shot470)
  (empty shot238)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker150 l0)
  (shaker-level shaker150 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail263 ingredient177)
  (cocktail-part2 cocktail263 ingredient248)
)
 (:goal
  (and
      (contains shot470 cocktail263)
)))
