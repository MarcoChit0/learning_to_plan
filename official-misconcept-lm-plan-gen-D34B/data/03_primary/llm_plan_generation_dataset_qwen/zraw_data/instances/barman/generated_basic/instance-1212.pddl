(define (problem prob)
 (:domain barman)
 (:objects 
      shaker451 - shaker
      left right - hand
      shot161 shot233 shot298 - shot
      ingredient142 ingredient6 ingredient34 - ingredient
      cocktail59 - cocktail
      dispenser359 dispenser327 dispenser299 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker451)
  (ontable shot161)
  (ontable shot233)
  (ontable shot298)
  (dispenses dispenser359 ingredient142)
  (dispenses dispenser327 ingredient6)
  (dispenses dispenser299 ingredient34)
  (clean shaker451)
  (clean shot161)
  (clean shot233)
  (clean shot298)
  (empty shaker451)
  (empty shot161)
  (empty shot233)
  (empty shot298)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker451 l0)
  (shaker-level shaker451 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail59 ingredient142)
  (cocktail-part2 cocktail59 ingredient34)
)
 (:goal
  (and
      (contains shot161 cocktail59)
      (contains shot233 cocktail59)
)))
