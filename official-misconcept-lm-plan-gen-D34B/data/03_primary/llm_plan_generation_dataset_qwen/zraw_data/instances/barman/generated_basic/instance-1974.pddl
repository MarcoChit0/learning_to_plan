(define (problem prob)
 (:domain barman)
 (:objects 
      shaker264 - shaker
      left right - hand
      shot69 shot411 - shot
      ingredient212 ingredient201 ingredient181 - ingredient
      cocktail468 - cocktail
      dispenser103 dispenser124 dispenser243 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker264)
  (ontable shot69)
  (ontable shot411)
  (dispenses dispenser103 ingredient212)
  (dispenses dispenser124 ingredient201)
  (dispenses dispenser243 ingredient181)
  (clean shaker264)
  (clean shot69)
  (clean shot411)
  (empty shaker264)
  (empty shot69)
  (empty shot411)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker264 l0)
  (shaker-level shaker264 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail468 ingredient201)
  (cocktail-part2 cocktail468 ingredient212)
)
 (:goal
  (and
      (contains shot69 cocktail468)
)))
