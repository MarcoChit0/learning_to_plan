(define (problem prob)
 (:domain barman)
 (:objects 
      shaker78 - shaker
      left right - hand
      shot301 - shot
      ingredient122 ingredient42 - ingredient
      cocktail267 - cocktail
      dispenser462 dispenser47 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker78)
  (ontable shot301)
  (dispenses dispenser462 ingredient122)
  (dispenses dispenser47 ingredient42)
  (clean shaker78)
  (clean shot301)
  (empty shaker78)
  (empty shot301)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker78 l0)
  (shaker-level shaker78 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail267 ingredient122)
  (cocktail-part2 cocktail267 ingredient42)
)
 (:goal
  (and
      (contains shot301 cocktail267)
)))
