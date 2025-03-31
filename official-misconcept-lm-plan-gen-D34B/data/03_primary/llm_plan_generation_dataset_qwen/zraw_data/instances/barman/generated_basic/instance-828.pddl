(define (problem prob)
 (:domain barman)
 (:objects 
      shaker465 - shaker
      left right - hand
      shot102 shot490 shot32 - shot
      ingredient271 ingredient152 ingredient52 ingredient321 - ingredient
      cocktail1 - cocktail
      dispenser224 dispenser349 dispenser327 dispenser217 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker465)
  (ontable shot102)
  (ontable shot490)
  (ontable shot32)
  (dispenses dispenser224 ingredient271)
  (dispenses dispenser349 ingredient152)
  (dispenses dispenser327 ingredient52)
  (dispenses dispenser217 ingredient321)
  (clean shaker465)
  (clean shot102)
  (clean shot490)
  (clean shot32)
  (empty shaker465)
  (empty shot102)
  (empty shot490)
  (empty shot32)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker465 l0)
  (shaker-level shaker465 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient321)
  (cocktail-part2 cocktail1 ingredient271)
)
 (:goal
  (and
      (contains shot102 cocktail1)
      (contains shot490 cocktail1)
)))
