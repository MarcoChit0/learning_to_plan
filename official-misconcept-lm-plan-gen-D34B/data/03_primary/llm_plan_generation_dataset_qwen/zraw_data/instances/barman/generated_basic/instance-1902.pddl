(define (problem prob)
 (:domain barman)
 (:objects 
      shaker43 - shaker
      left right - hand
      shot315 shot433 shot345 shot12 - shot
      ingredient403 ingredient240 - ingredient
      cocktail235 - cocktail
      dispenser10 dispenser72 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker43)
  (ontable shot315)
  (ontable shot433)
  (ontable shot345)
  (ontable shot12)
  (dispenses dispenser10 ingredient403)
  (dispenses dispenser72 ingredient240)
  (clean shaker43)
  (clean shot315)
  (clean shot433)
  (clean shot345)
  (clean shot12)
  (empty shaker43)
  (empty shot315)
  (empty shot433)
  (empty shot345)
  (empty shot12)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker43 l0)
  (shaker-level shaker43 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail235 ingredient403)
  (cocktail-part2 cocktail235 ingredient240)
)
 (:goal
  (and
      (contains shot315 cocktail235)
      (contains shot433 cocktail235)
      (contains shot345 ingredient403)
)))
