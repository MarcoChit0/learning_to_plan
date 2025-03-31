(define (problem prob)
 (:domain barman)
 (:objects 
      shaker462 - shaker
      left right - hand
      shot444 shot3 shot494 - shot
      ingredient491 ingredient273 ingredient366 ingredient123 - ingredient
      cocktail158 - cocktail
      dispenser204 dispenser164 dispenser311 dispenser91 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker462)
  (ontable shot444)
  (ontable shot3)
  (ontable shot494)
  (dispenses dispenser204 ingredient491)
  (dispenses dispenser164 ingredient273)
  (dispenses dispenser311 ingredient366)
  (dispenses dispenser91 ingredient123)
  (clean shaker462)
  (clean shot444)
  (clean shot3)
  (clean shot494)
  (empty shaker462)
  (empty shot444)
  (empty shot3)
  (empty shot494)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker462 l0)
  (shaker-level shaker462 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail158 ingredient123)
  (cocktail-part2 cocktail158 ingredient273)
)
 (:goal
  (and
      (contains shot444 cocktail158)
      (contains shot3 cocktail158)
)))
