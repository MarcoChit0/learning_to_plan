(define (problem prob)
 (:domain barman)
 (:objects 
      shaker324 - shaker
      left right - hand
      shot405 shot114 shot245 - shot
      ingredient156 ingredient146 - ingredient
      cocktail1 - cocktail
      dispenser85 dispenser224 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker324)
  (ontable shot405)
  (ontable shot114)
  (ontable shot245)
  (dispenses dispenser85 ingredient156)
  (dispenses dispenser224 ingredient146)
  (clean shaker324)
  (clean shot405)
  (clean shot114)
  (clean shot245)
  (empty shaker324)
  (empty shot405)
  (empty shot114)
  (empty shot245)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker324 l0)
  (shaker-level shaker324 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient156)
  (cocktail-part2 cocktail1 ingredient146)
)
 (:goal
  (and
      (contains shot405 cocktail1)
      (contains shot114 ingredient146)
)))
