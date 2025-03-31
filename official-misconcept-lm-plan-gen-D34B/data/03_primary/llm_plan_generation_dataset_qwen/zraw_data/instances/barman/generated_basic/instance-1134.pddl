(define (problem prob)
 (:domain barman)
 (:objects 
      shaker301 - shaker
      left right - hand
      shot284 shot107 - shot
      ingredient387 ingredient455 ingredient188 - ingredient
      cocktail1 - cocktail
      dispenser63 dispenser105 dispenser99 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker301)
  (ontable shot284)
  (ontable shot107)
  (dispenses dispenser63 ingredient387)
  (dispenses dispenser105 ingredient455)
  (dispenses dispenser99 ingredient188)
  (clean shaker301)
  (clean shot284)
  (clean shot107)
  (empty shaker301)
  (empty shot284)
  (empty shot107)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker301 l0)
  (shaker-level shaker301 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient387)
  (cocktail-part2 cocktail1 ingredient455)
)
 (:goal
  (and
      (contains shot284 cocktail1)
)))
