(define (problem prob)
 (:domain barman)
 (:objects 
      shaker464 - shaker
      left right - hand
      shot374 shot132 shot264 - shot
      ingredient264 ingredient396 - ingredient
      cocktail374 - cocktail
      dispenser339 dispenser150 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker464)
  (ontable shot374)
  (ontable shot132)
  (ontable shot264)
  (dispenses dispenser339 ingredient264)
  (dispenses dispenser150 ingredient396)
  (clean shaker464)
  (clean shot374)
  (clean shot132)
  (clean shot264)
  (empty shaker464)
  (empty shot374)
  (empty shot132)
  (empty shot264)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker464 l0)
  (shaker-level shaker464 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail374 ingredient264)
  (cocktail-part2 cocktail374 ingredient396)
)
 (:goal
  (and
      (contains shot374 cocktail374)
      (contains shot132 ingredient396)
)))
