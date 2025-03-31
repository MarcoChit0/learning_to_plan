(define (problem prob)
 (:domain barman)
 (:objects 
      shaker446 - shaker
      left right - hand
      shot264 - shot
      ingredient289 ingredient192 ingredient74 - ingredient
      cocktail266 - cocktail
      dispenser150 dispenser266 dispenser387 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker446)
  (ontable shot264)
  (dispenses dispenser150 ingredient289)
  (dispenses dispenser266 ingredient192)
  (dispenses dispenser387 ingredient74)
  (clean shaker446)
  (clean shot264)
  (empty shaker446)
  (empty shot264)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker446 l0)
  (shaker-level shaker446 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail266 ingredient192)
  (cocktail-part2 cocktail266 ingredient289)
)
 (:goal
  (and
      (contains shot264 cocktail266)
)))
