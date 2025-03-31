(define (problem prob)
 (:domain barman)
 (:objects 
      shaker335 - shaker
      left right - hand
      shot291 shot215 - shot
      ingredient140 ingredient24 ingredient284 ingredient194 - ingredient
      cocktail338 - cocktail
      dispenser311 dispenser23 dispenser142 dispenser58 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker335)
  (ontable shot291)
  (ontable shot215)
  (dispenses dispenser311 ingredient140)
  (dispenses dispenser23 ingredient24)
  (dispenses dispenser142 ingredient284)
  (dispenses dispenser58 ingredient194)
  (clean shaker335)
  (clean shot291)
  (clean shot215)
  (empty shaker335)
  (empty shot291)
  (empty shot215)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker335 l0)
  (shaker-level shaker335 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail338 ingredient24)
  (cocktail-part2 cocktail338 ingredient284)
)
 (:goal
  (and
      (contains shot291 cocktail338)
)))
