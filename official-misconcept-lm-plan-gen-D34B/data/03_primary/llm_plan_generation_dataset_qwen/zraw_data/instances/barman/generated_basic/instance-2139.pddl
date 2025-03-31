(define (problem prob)
 (:domain barman)
 (:objects 
      shaker315 - shaker
      left right - hand
      shot14 shot36 shot144 - shot
      ingredient264 ingredient364 - ingredient
      cocktail14 - cocktail
      dispenser153 dispenser58 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker315)
  (ontable shot14)
  (ontable shot36)
  (ontable shot144)
  (dispenses dispenser153 ingredient264)
  (dispenses dispenser58 ingredient364)
  (clean shaker315)
  (clean shot14)
  (clean shot36)
  (clean shot144)
  (empty shaker315)
  (empty shot14)
  (empty shot36)
  (empty shot144)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker315 l0)
  (shaker-level shaker315 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail14 ingredient364)
  (cocktail-part2 cocktail14 ingredient264)
)
 (:goal
  (and
      (contains shot14 cocktail14)
      (contains shot36 cocktail14)
)))
