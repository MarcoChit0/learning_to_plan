(define (problem prob)
 (:domain barman)
 (:objects 
      shaker83 - shaker
      left right - hand
      shot111 shot133 shot432 - shot
      ingredient34 ingredient301 - ingredient
      cocktail317 - cocktail
      dispenser330 dispenser348 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker83)
  (ontable shot111)
  (ontable shot133)
  (ontable shot432)
  (dispenses dispenser330 ingredient34)
  (dispenses dispenser348 ingredient301)
  (clean shaker83)
  (clean shot111)
  (clean shot133)
  (clean shot432)
  (empty shaker83)
  (empty shot111)
  (empty shot133)
  (empty shot432)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker83 l0)
  (shaker-level shaker83 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail317 ingredient34)
  (cocktail-part2 cocktail317 ingredient301)
)
 (:goal
  (and
      (contains shot111 cocktail317)
      (contains shot133 ingredient301)
)))
