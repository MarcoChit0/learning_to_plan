(define (problem prob)
 (:domain barman)
 (:objects 
      shaker264 - shaker
      left right - hand
      shot134 - shot
      ingredient155 ingredient73 ingredient70 ingredient331 - ingredient
      cocktail59 - cocktail
      dispenser78 dispenser379 dispenser137 dispenser310 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker264)
  (ontable shot134)
  (dispenses dispenser78 ingredient155)
  (dispenses dispenser379 ingredient73)
  (dispenses dispenser137 ingredient70)
  (dispenses dispenser310 ingredient331)
  (clean shaker264)
  (clean shot134)
  (empty shaker264)
  (empty shot134)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker264 l0)
  (shaker-level shaker264 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail59 ingredient155)
  (cocktail-part2 cocktail59 ingredient70)
)
 (:goal
  (and
      (contains shot134 cocktail59)
)))
