(define (problem prob)
 (:domain barman)
 (:objects 
      shaker306 - shaker
      left right - hand
      shot174 shot249 - shot
      ingredient173 ingredient442 ingredient212 - ingredient
      cocktail1 - cocktail
      dispenser475 dispenser227 dispenser50 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker306)
  (ontable shot174)
  (ontable shot249)
  (dispenses dispenser475 ingredient173)
  (dispenses dispenser227 ingredient442)
  (dispenses dispenser50 ingredient212)
  (clean shaker306)
  (clean shot174)
  (clean shot249)
  (empty shaker306)
  (empty shot174)
  (empty shot249)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker306 l0)
  (shaker-level shaker306 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient212)
  (cocktail-part2 cocktail1 ingredient442)
)
 (:goal
  (and
      (contains shot174 cocktail1)
)))
