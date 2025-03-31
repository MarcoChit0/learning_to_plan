(define (problem prob)
 (:domain barman)
 (:objects 
      shaker264 - shaker
      left right - hand
      shot221 shot281 shot325 - shot
      ingredient183 ingredient156 - ingredient
      cocktail16 - cocktail
      dispenser58 dispenser134 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker264)
  (ontable shot221)
  (ontable shot281)
  (ontable shot325)
  (dispenses dispenser58 ingredient183)
  (dispenses dispenser134 ingredient156)
  (clean shaker264)
  (clean shot221)
  (clean shot281)
  (clean shot325)
  (empty shaker264)
  (empty shot221)
  (empty shot281)
  (empty shot325)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker264 l0)
  (shaker-level shaker264 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail16 ingredient183)
  (cocktail-part2 cocktail16 ingredient156)
)
 (:goal
  (and
      (contains shot221 cocktail16)
      (contains shot281 cocktail16)
)))
