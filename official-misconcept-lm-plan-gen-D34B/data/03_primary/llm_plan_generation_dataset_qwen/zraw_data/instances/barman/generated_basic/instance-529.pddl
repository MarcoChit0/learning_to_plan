(define (problem prob)
 (:domain barman)
 (:objects 
      shaker235 - shaker
      left right - hand
      shot85 shot449 - shot
      ingredient35 ingredient96 - ingredient
      cocktail1 - cocktail
      dispenser349 dispenser311 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker235)
  (ontable shot85)
  (ontable shot449)
  (dispenses dispenser349 ingredient35)
  (dispenses dispenser311 ingredient96)
  (clean shaker235)
  (clean shot85)
  (clean shot449)
  (empty shaker235)
  (empty shot85)
  (empty shot449)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker235 l0)
  (shaker-level shaker235 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient96)
  (cocktail-part2 cocktail1 ingredient35)
)
 (:goal
  (and
      (contains shot85 cocktail1)
)))
