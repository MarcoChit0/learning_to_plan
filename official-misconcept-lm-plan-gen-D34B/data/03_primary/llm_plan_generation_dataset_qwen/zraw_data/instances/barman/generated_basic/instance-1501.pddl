(define (problem prob)
 (:domain barman)
 (:objects 
      shaker66 - shaker
      left right - hand
      shot239 shot443 - shot
      ingredient490 ingredient82 ingredient58 ingredient348 - ingredient
      cocktail234 - cocktail
      dispenser57 dispenser139 dispenser193 dispenser99 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker66)
  (ontable shot239)
  (ontable shot443)
  (dispenses dispenser57 ingredient490)
  (dispenses dispenser139 ingredient82)
  (dispenses dispenser193 ingredient58)
  (dispenses dispenser99 ingredient348)
  (clean shaker66)
  (clean shot239)
  (clean shot443)
  (empty shaker66)
  (empty shot239)
  (empty shot443)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker66 l0)
  (shaker-level shaker66 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail234 ingredient490)
  (cocktail-part2 cocktail234 ingredient82)
)
 (:goal
  (and
      (contains shot239 cocktail234)
)))
