(define (problem prob)
 (:domain barman)
 (:objects 
      shaker122 - shaker
      left right - hand
      shot89 shot162 - shot
      ingredient489 ingredient86 ingredient264 - ingredient
      cocktail1 - cocktail
      dispenser400 dispenser9 dispenser132 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker122)
  (ontable shot89)
  (ontable shot162)
  (dispenses dispenser400 ingredient489)
  (dispenses dispenser9 ingredient86)
  (dispenses dispenser132 ingredient264)
  (clean shaker122)
  (clean shot89)
  (clean shot162)
  (empty shaker122)
  (empty shot89)
  (empty shot162)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker122 l0)
  (shaker-level shaker122 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient86)
  (cocktail-part2 cocktail1 ingredient264)
)
 (:goal
  (and
      (contains shot89 cocktail1)
)))
