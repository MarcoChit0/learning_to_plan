(define (problem prob)
 (:domain barman)
 (:objects 
      shaker136 - shaker
      left right - hand
      shot215 shot114 shot74 - shot
      ingredient37 ingredient31 ingredient125 ingredient63 - ingredient
      cocktail358 - cocktail
      dispenser228 dispenser287 dispenser62 dispenser4 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker136)
  (ontable shot215)
  (ontable shot114)
  (ontable shot74)
  (dispenses dispenser228 ingredient37)
  (dispenses dispenser287 ingredient31)
  (dispenses dispenser62 ingredient125)
  (dispenses dispenser4 ingredient63)
  (clean shaker136)
  (clean shot215)
  (clean shot114)
  (clean shot74)
  (empty shaker136)
  (empty shot215)
  (empty shot114)
  (empty shot74)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker136 l0)
  (shaker-level shaker136 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail358 ingredient125)
  (cocktail-part2 cocktail358 ingredient37)
)
 (:goal
  (and
      (contains shot215 cocktail358)
      (contains shot114 cocktail358)
)))
