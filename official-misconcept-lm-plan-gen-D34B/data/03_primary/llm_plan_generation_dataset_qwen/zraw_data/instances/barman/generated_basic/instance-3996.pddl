(define (problem prob)
 (:domain barman)
 (:objects 
      shaker99 - shaker
      left right - hand
      shot320 shot301 shot440 - shot
      ingredient399 ingredient158 - ingredient
      cocktail215 - cocktail
      dispenser240 dispenser321 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker99)
  (ontable shot320)
  (ontable shot301)
  (ontable shot440)
  (dispenses dispenser240 ingredient399)
  (dispenses dispenser321 ingredient158)
  (clean shaker99)
  (clean shot320)
  (clean shot301)
  (clean shot440)
  (empty shaker99)
  (empty shot320)
  (empty shot301)
  (empty shot440)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker99 l0)
  (shaker-level shaker99 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail215 ingredient158)
  (cocktail-part2 cocktail215 ingredient399)
)
 (:goal
  (and
      (contains shot320 cocktail215)
      (contains shot301 ingredient399)
)))
