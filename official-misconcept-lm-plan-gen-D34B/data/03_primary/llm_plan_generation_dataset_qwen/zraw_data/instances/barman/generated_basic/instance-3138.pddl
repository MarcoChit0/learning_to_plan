(define (problem prob)
 (:domain barman)
 (:objects 
      shaker10 - shaker
      left right - hand
      shot320 shot381 shot145 - shot
      ingredient493 ingredient312 - ingredient
      cocktail264 - cocktail
      dispenser148 dispenser4 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker10)
  (ontable shot320)
  (ontable shot381)
  (ontable shot145)
  (dispenses dispenser148 ingredient493)
  (dispenses dispenser4 ingredient312)
  (clean shaker10)
  (clean shot320)
  (clean shot381)
  (clean shot145)
  (empty shaker10)
  (empty shot320)
  (empty shot381)
  (empty shot145)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker10 l0)
  (shaker-level shaker10 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail264 ingredient312)
  (cocktail-part2 cocktail264 ingredient493)
)
 (:goal
  (and
      (contains shot320 cocktail264)
      (contains shot381 cocktail264)
)))
