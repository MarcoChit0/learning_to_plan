(define (problem prob)
 (:domain barman)
 (:objects 
      shaker386 - shaker
      left right - hand
      shot274 shot154 shot338 - shot
      ingredient167 ingredient389 ingredient331 - ingredient
      cocktail46 - cocktail
      dispenser273 dispenser399 dispenser220 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker386)
  (ontable shot274)
  (ontable shot154)
  (ontable shot338)
  (dispenses dispenser273 ingredient167)
  (dispenses dispenser399 ingredient389)
  (dispenses dispenser220 ingredient331)
  (clean shaker386)
  (clean shot274)
  (clean shot154)
  (clean shot338)
  (empty shaker386)
  (empty shot274)
  (empty shot154)
  (empty shot338)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker386 l0)
  (shaker-level shaker386 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail46 ingredient389)
  (cocktail-part2 cocktail46 ingredient167)
)
 (:goal
  (and
      (contains shot274 cocktail46)
      (contains shot154 cocktail46)
)))
