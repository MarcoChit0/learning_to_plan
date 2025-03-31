(define (problem prob)
 (:domain barman)
 (:objects 
      shaker238 - shaker
      left right - hand
      shot467 shot222 - shot
      ingredient326 ingredient364 - ingredient
      cocktail317 - cocktail
      dispenser25 dispenser261 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker238)
  (ontable shot467)
  (ontable shot222)
  (dispenses dispenser25 ingredient326)
  (dispenses dispenser261 ingredient364)
  (clean shaker238)
  (clean shot467)
  (clean shot222)
  (empty shaker238)
  (empty shot467)
  (empty shot222)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker238 l0)
  (shaker-level shaker238 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail317 ingredient364)
  (cocktail-part2 cocktail317 ingredient326)
)
 (:goal
  (and
      (contains shot467 cocktail317)
)))
