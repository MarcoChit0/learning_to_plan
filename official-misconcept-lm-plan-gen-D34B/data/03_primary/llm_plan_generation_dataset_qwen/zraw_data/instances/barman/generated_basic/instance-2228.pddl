(define (problem prob)
 (:domain barman)
 (:objects 
      shaker234 - shaker
      left right - hand
      shot78 - shot
      ingredient399 ingredient134 - ingredient
      cocktail368 - cocktail
      dispenser9 dispenser272 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker234)
  (ontable shot78)
  (dispenses dispenser9 ingredient399)
  (dispenses dispenser272 ingredient134)
  (clean shaker234)
  (clean shot78)
  (empty shaker234)
  (empty shot78)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker234 l0)
  (shaker-level shaker234 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail368 ingredient134)
  (cocktail-part2 cocktail368 ingredient399)
)
 (:goal
  (and
      (contains shot78 cocktail368)
)))
