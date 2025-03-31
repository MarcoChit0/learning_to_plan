(define (problem prob)
 (:domain barman)
 (:objects 
      shaker130 - shaker
      left right - hand
      shot402 shot17 - shot
      ingredient275 ingredient264 ingredient12 - ingredient
      cocktail1 - cocktail
      dispenser217 dispenser101 dispenser350 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker130)
  (ontable shot402)
  (ontable shot17)
  (dispenses dispenser217 ingredient275)
  (dispenses dispenser101 ingredient264)
  (dispenses dispenser350 ingredient12)
  (clean shaker130)
  (clean shot402)
  (clean shot17)
  (empty shaker130)
  (empty shot402)
  (empty shot17)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker130 l0)
  (shaker-level shaker130 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient264)
  (cocktail-part2 cocktail1 ingredient12)
)
 (:goal
  (and
      (contains shot402 cocktail1)
)))
