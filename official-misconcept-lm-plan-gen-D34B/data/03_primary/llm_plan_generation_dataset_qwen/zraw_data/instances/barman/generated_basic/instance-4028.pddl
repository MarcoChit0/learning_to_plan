(define (problem prob)
 (:domain barman)
 (:objects 
      shaker275 - shaker
      left right - hand
      shot489 shot70 - shot
      ingredient26 ingredient264 ingredient168 - ingredient
      cocktail469 - cocktail
      dispenser472 dispenser276 dispenser150 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker275)
  (ontable shot489)
  (ontable shot70)
  (dispenses dispenser472 ingredient26)
  (dispenses dispenser276 ingredient264)
  (dispenses dispenser150 ingredient168)
  (clean shaker275)
  (clean shot489)
  (clean shot70)
  (empty shaker275)
  (empty shot489)
  (empty shot70)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker275 l0)
  (shaker-level shaker275 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail469 ingredient264)
  (cocktail-part2 cocktail469 ingredient26)
)
 (:goal
  (and
      (contains shot489 cocktail469)
)))
