(define (problem prob)
 (:domain barman)
 (:objects 
      shaker32 - shaker
      left right - hand
      shot99 shot154 shot226 - shot
      ingredient417 ingredient305 - ingredient
      cocktail271 - cocktail
      dispenser312 dispenser395 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker32)
  (ontable shot99)
  (ontable shot154)
  (ontable shot226)
  (dispenses dispenser312 ingredient417)
  (dispenses dispenser395 ingredient305)
  (clean shaker32)
  (clean shot99)
  (clean shot154)
  (clean shot226)
  (empty shaker32)
  (empty shot99)
  (empty shot154)
  (empty shot226)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker32 l0)
  (shaker-level shaker32 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail271 ingredient417)
  (cocktail-part2 cocktail271 ingredient305)
)
 (:goal
  (and
      (contains shot99 cocktail271)
      (contains shot154 ingredient305)
)))
