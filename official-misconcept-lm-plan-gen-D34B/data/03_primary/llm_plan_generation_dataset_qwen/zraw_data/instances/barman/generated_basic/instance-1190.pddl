(define (problem prob)
 (:domain barman)
 (:objects 
      shaker380 - shaker
      left right - hand
      shot233 shot449 - shot
      ingredient447 ingredient264 - ingredient
      cocktail162 - cocktail
      dispenser490 dispenser313 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker380)
  (ontable shot233)
  (ontable shot449)
  (dispenses dispenser490 ingredient447)
  (dispenses dispenser313 ingredient264)
  (clean shaker380)
  (clean shot233)
  (clean shot449)
  (empty shaker380)
  (empty shot233)
  (empty shot449)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker380 l0)
  (shaker-level shaker380 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail162 ingredient447)
  (cocktail-part2 cocktail162 ingredient264)
)
 (:goal
  (and
      (contains shot233 cocktail162)
)))
