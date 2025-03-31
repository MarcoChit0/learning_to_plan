(define (problem prob)
 (:domain barman)
 (:objects 
      shaker183 - shaker
      left right - hand
      shot299 shot264 - shot
      ingredient70 ingredient348 ingredient366 ingredient270 - ingredient
      cocktail14 - cocktail
      dispenser448 dispenser234 dispenser224 dispenser372 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker183)
  (ontable shot299)
  (ontable shot264)
  (dispenses dispenser448 ingredient70)
  (dispenses dispenser234 ingredient348)
  (dispenses dispenser224 ingredient366)
  (dispenses dispenser372 ingredient270)
  (clean shaker183)
  (clean shot299)
  (clean shot264)
  (empty shaker183)
  (empty shot299)
  (empty shot264)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker183 l0)
  (shaker-level shaker183 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail14 ingredient366)
  (cocktail-part2 cocktail14 ingredient270)
)
 (:goal
  (and
      (contains shot299 cocktail14)
)))
