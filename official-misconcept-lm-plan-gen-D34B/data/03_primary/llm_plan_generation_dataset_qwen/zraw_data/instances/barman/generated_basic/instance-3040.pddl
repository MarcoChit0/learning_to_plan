(define (problem prob)
 (:domain barman)
 (:objects 
      shaker249 - shaker
      left right - hand
      shot59 shot484 - shot
      ingredient493 ingredient420 ingredient153 - ingredient
      cocktail16 - cocktail
      dispenser260 dispenser327 dispenser308 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker249)
  (ontable shot59)
  (ontable shot484)
  (dispenses dispenser260 ingredient493)
  (dispenses dispenser327 ingredient420)
  (dispenses dispenser308 ingredient153)
  (clean shaker249)
  (clean shot59)
  (clean shot484)
  (empty shaker249)
  (empty shot59)
  (empty shot484)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker249 l0)
  (shaker-level shaker249 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail16 ingredient420)
  (cocktail-part2 cocktail16 ingredient493)
)
 (:goal
  (and
      (contains shot59 cocktail16)
)))
