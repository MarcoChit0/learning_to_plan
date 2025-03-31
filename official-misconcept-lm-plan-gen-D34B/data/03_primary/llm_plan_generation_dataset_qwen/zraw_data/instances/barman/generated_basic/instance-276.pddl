(define (problem prob)
 (:domain barman)
 (:objects 
      shaker301 - shaker
      left right - hand
      shot399 shot490 - shot
      ingredient97 ingredient73 ingredient492 - ingredient
      cocktail1 - cocktail
      dispenser428 dispenser186 dispenser458 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker301)
  (ontable shot399)
  (ontable shot490)
  (dispenses dispenser428 ingredient97)
  (dispenses dispenser186 ingredient73)
  (dispenses dispenser458 ingredient492)
  (clean shaker301)
  (clean shot399)
  (clean shot490)
  (empty shaker301)
  (empty shot399)
  (empty shot490)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker301 l0)
  (shaker-level shaker301 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient73)
  (cocktail-part2 cocktail1 ingredient492)
)
 (:goal
  (and
      (contains shot399 cocktail1)
)))
