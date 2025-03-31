(define (problem prob)
 (:domain barman)
 (:objects 
      shaker23 - shaker
      left right - hand
      shot120 shot287 - shot
      ingredient60 ingredient400 ingredient432 ingredient264 - ingredient
      cocktail6 - cocktail
      dispenser124 dispenser40 dispenser180 dispenser232 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker23)
  (ontable shot120)
  (ontable shot287)
  (dispenses dispenser124 ingredient60)
  (dispenses dispenser40 ingredient400)
  (dispenses dispenser180 ingredient432)
  (dispenses dispenser232 ingredient264)
  (clean shaker23)
  (clean shot120)
  (clean shot287)
  (empty shaker23)
  (empty shot120)
  (empty shot287)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker23 l0)
  (shaker-level shaker23 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail6 ingredient60)
  (cocktail-part2 cocktail6 ingredient432)
)
 (:goal
  (and
      (contains shot120 cocktail6)
)))
