(define (problem prob)
 (:domain barman)
 (:objects 
      shaker331 - shaker
      left right - hand
      shot374 shot43 shot279 - shot
      ingredient227 ingredient240 ingredient313 ingredient479 - ingredient
      cocktail210 - cocktail
      dispenser490 dispenser144 dispenser58 dispenser370 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker331)
  (ontable shot374)
  (ontable shot43)
  (ontable shot279)
  (dispenses dispenser490 ingredient227)
  (dispenses dispenser144 ingredient240)
  (dispenses dispenser58 ingredient313)
  (dispenses dispenser370 ingredient479)
  (clean shaker331)
  (clean shot374)
  (clean shot43)
  (clean shot279)
  (empty shaker331)
  (empty shot374)
  (empty shot43)
  (empty shot279)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker331 l0)
  (shaker-level shaker331 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail210 ingredient313)
  (cocktail-part2 cocktail210 ingredient240)
)
 (:goal
  (and
      (contains shot374 cocktail210)
      (contains shot43 ingredient479)
)))
