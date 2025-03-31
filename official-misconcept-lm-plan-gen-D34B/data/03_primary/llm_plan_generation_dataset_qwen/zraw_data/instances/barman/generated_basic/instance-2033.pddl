(define (problem prob)
 (:domain barman)
 (:objects 
      shaker423 - shaker
      left right - hand
      shot222 shot176 - shot
      ingredient280 ingredient317 - ingredient
      cocktail417 - cocktail
      dispenser36 dispenser99 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker423)
  (ontable shot222)
  (ontable shot176)
  (dispenses dispenser36 ingredient280)
  (dispenses dispenser99 ingredient317)
  (clean shaker423)
  (clean shot222)
  (clean shot176)
  (empty shaker423)
  (empty shot222)
  (empty shot176)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker423 l0)
  (shaker-level shaker423 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail417 ingredient317)
  (cocktail-part2 cocktail417 ingredient280)
)
 (:goal
  (and
      (contains shot222 cocktail417)
)))
