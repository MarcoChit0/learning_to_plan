(define (problem prob)
 (:domain barman)
 (:objects 
      shaker260 - shaker
      left right - hand
      shot411 shot485 shot149 - shot
      ingredient264 ingredient75 ingredient129 ingredient32 - ingredient
      cocktail1 - cocktail
      dispenser263 dispenser266 dispenser22 dispenser340 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker260)
  (ontable shot411)
  (ontable shot485)
  (ontable shot149)
  (dispenses dispenser263 ingredient264)
  (dispenses dispenser266 ingredient75)
  (dispenses dispenser22 ingredient129)
  (dispenses dispenser340 ingredient32)
  (clean shaker260)
  (clean shot411)
  (clean shot485)
  (clean shot149)
  (empty shaker260)
  (empty shot411)
  (empty shot485)
  (empty shot149)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker260 l0)
  (shaker-level shaker260 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient264)
  (cocktail-part2 cocktail1 ingredient75)
)
 (:goal
  (and
      (contains shot411 cocktail1)
      (contains shot485 cocktail1)
)))
