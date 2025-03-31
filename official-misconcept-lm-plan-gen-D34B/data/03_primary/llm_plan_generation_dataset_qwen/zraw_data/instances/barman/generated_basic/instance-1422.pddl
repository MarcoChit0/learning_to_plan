(define (problem prob)
 (:domain barman)
 (:objects 
      shaker411 - shaker
      left right - hand
      shot151 shot264 - shot
      ingredient411 ingredient99 - ingredient
      cocktail153 - cocktail
      dispenser371 dispenser336 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker411)
  (ontable shot151)
  (ontable shot264)
  (dispenses dispenser371 ingredient411)
  (dispenses dispenser336 ingredient99)
  (clean shaker411)
  (clean shot151)
  (clean shot264)
  (empty shaker411)
  (empty shot151)
  (empty shot264)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker411 l0)
  (shaker-level shaker411 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail153 ingredient411)
  (cocktail-part2 cocktail153 ingredient99)
)
 (:goal
  (and
      (contains shot151 cocktail153)
)))
