(define (problem prob)
 (:domain barman)
 (:objects 
      shaker315 - shaker
      left right - hand
      shot188 shot300 shot403 - shot
      ingredient377 ingredient266 - ingredient
      cocktail370 - cocktail
      dispenser349 dispenser193 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker315)
  (ontable shot188)
  (ontable shot300)
  (ontable shot403)
  (dispenses dispenser349 ingredient377)
  (dispenses dispenser193 ingredient266)
  (clean shaker315)
  (clean shot188)
  (clean shot300)
  (clean shot403)
  (empty shaker315)
  (empty shot188)
  (empty shot300)
  (empty shot403)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker315 l0)
  (shaker-level shaker315 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail370 ingredient266)
  (cocktail-part2 cocktail370 ingredient377)
)
 (:goal
  (and
      (contains shot188 cocktail370)
      (contains shot300 cocktail370)
)))
