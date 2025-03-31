(define (problem prob)
 (:domain barman)
 (:objects 
      shaker8 - shaker
      left right - hand
      shot298 shot363 shot239 - shot
      ingredient441 ingredient264 - ingredient
      cocktail183 - cocktail
      dispenser38 dispenser85 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker8)
  (ontable shot298)
  (ontable shot363)
  (ontable shot239)
  (dispenses dispenser38 ingredient441)
  (dispenses dispenser85 ingredient264)
  (clean shaker8)
  (clean shot298)
  (clean shot363)
  (clean shot239)
  (empty shaker8)
  (empty shot298)
  (empty shot363)
  (empty shot239)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker8 l0)
  (shaker-level shaker8 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail183 ingredient264)
  (cocktail-part2 cocktail183 ingredient441)
)
 (:goal
  (and
      (contains shot298 cocktail183)
      (contains shot363 cocktail183)
)))
