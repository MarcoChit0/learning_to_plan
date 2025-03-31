(define (problem prob)
 (:domain barman)
 (:objects 
      shaker31 - shaker
      left right - hand
      shot78 shot301 - shot
      ingredient120 ingredient228 ingredient192 - ingredient
      cocktail46 - cocktail
      dispenser425 dispenser328 dispenser347 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker31)
  (ontable shot78)
  (ontable shot301)
  (dispenses dispenser425 ingredient120)
  (dispenses dispenser328 ingredient228)
  (dispenses dispenser347 ingredient192)
  (clean shaker31)
  (clean shot78)
  (clean shot301)
  (empty shaker31)
  (empty shot78)
  (empty shot301)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker31 l0)
  (shaker-level shaker31 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail46 ingredient192)
  (cocktail-part2 cocktail46 ingredient228)
)
 (:goal
  (and
      (contains shot78 cocktail46)
)))
