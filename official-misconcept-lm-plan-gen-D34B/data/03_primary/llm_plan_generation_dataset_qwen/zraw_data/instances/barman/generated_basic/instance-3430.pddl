(define (problem prob)
 (:domain barman)
 (:objects 
      shaker233 - shaker
      left right - hand
      shot45 shot228 - shot
      ingredient264 ingredient243 ingredient482 - ingredient
      cocktail156 - cocktail
      dispenser219 dispenser244 dispenser257 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker233)
  (ontable shot45)
  (ontable shot228)
  (dispenses dispenser219 ingredient264)
  (dispenses dispenser244 ingredient243)
  (dispenses dispenser257 ingredient482)
  (clean shaker233)
  (clean shot45)
  (clean shot228)
  (empty shaker233)
  (empty shot45)
  (empty shot228)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker233 l0)
  (shaker-level shaker233 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail156 ingredient482)
  (cocktail-part2 cocktail156 ingredient243)
)
 (:goal
  (and
      (contains shot45 cocktail156)
)))
