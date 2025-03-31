(define (problem prob)
 (:domain barman)
 (:objects 
      shaker264 - shaker
      left right - hand
      shot454 shot37 - shot
      ingredient263 ingredient83 ingredient322 ingredient4 - ingredient
      cocktail1 - cocktail
      dispenser354 dispenser182 dispenser213 dispenser117 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker264)
  (ontable shot454)
  (ontable shot37)
  (dispenses dispenser354 ingredient263)
  (dispenses dispenser182 ingredient83)
  (dispenses dispenser213 ingredient322)
  (dispenses dispenser117 ingredient4)
  (clean shaker264)
  (clean shot454)
  (clean shot37)
  (empty shaker264)
  (empty shot454)
  (empty shot37)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker264 l0)
  (shaker-level shaker264 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient83)
  (cocktail-part2 cocktail1 ingredient4)
)
 (:goal
  (and
      (contains shot454 cocktail1)
)))
