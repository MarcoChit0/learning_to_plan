(define (problem prob)
 (:domain barman)
 (:objects 
      shaker377 - shaker
      left right - hand
      shot250 shot368 - shot
      ingredient301 ingredient65 - ingredient
      cocktail1 - cocktail
      dispenser433 dispenser244 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker377)
  (ontable shot250)
  (ontable shot368)
  (dispenses dispenser433 ingredient301)
  (dispenses dispenser244 ingredient65)
  (clean shaker377)
  (clean shot250)
  (clean shot368)
  (empty shaker377)
  (empty shot250)
  (empty shot368)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker377 l0)
  (shaker-level shaker377 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient65)
  (cocktail-part2 cocktail1 ingredient301)
)
 (:goal
  (and
      (contains shot250 cocktail1)
)))
