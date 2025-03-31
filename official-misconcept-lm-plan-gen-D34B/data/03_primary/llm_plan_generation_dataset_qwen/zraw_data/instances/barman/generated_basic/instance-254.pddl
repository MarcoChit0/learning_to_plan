(define (problem prob)
 (:domain barman)
 (:objects 
      shaker24 - shaker
      left right - hand
      shot152 shot295 shot369 - shot
      ingredient67 ingredient354 ingredient121 - ingredient
      cocktail1 - cocktail
      dispenser489 dispenser74 dispenser357 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker24)
  (ontable shot152)
  (ontable shot295)
  (ontable shot369)
  (dispenses dispenser489 ingredient67)
  (dispenses dispenser74 ingredient354)
  (dispenses dispenser357 ingredient121)
  (clean shaker24)
  (clean shot152)
  (clean shot295)
  (clean shot369)
  (empty shaker24)
  (empty shot152)
  (empty shot295)
  (empty shot369)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker24 l0)
  (shaker-level shaker24 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient121)
  (cocktail-part2 cocktail1 ingredient67)
)
 (:goal
  (and
      (contains shot152 cocktail1)
      (contains shot295 cocktail1)
)))
