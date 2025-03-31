(define (problem prob)
 (:domain barman)
 (:objects 
      shaker375 - shaker
      left right - hand
      shot441 shot339 shot294 - shot
      ingredient478 ingredient314 ingredient300 - ingredient
      cocktail329 - cocktail
      dispenser442 dispenser161 dispenser138 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker375)
  (ontable shot441)
  (ontable shot339)
  (ontable shot294)
  (dispenses dispenser442 ingredient478)
  (dispenses dispenser161 ingredient314)
  (dispenses dispenser138 ingredient300)
  (clean shaker375)
  (clean shot441)
  (clean shot339)
  (clean shot294)
  (empty shaker375)
  (empty shot441)
  (empty shot339)
  (empty shot294)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker375 l0)
  (shaker-level shaker375 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail329 ingredient478)
  (cocktail-part2 cocktail329 ingredient300)
)
 (:goal
  (and
      (contains shot441 cocktail329)
      (contains shot339 ingredient300)
)))
