(define (problem prob)
 (:domain barman)
 (:objects 
      shaker67 - shaker
      left right - hand
      shot139 shot438 - shot
      ingredient489 ingredient294 - ingredient
      cocktail32 - cocktail
      dispenser218 dispenser21 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker67)
  (ontable shot139)
  (ontable shot438)
  (dispenses dispenser218 ingredient489)
  (dispenses dispenser21 ingredient294)
  (clean shaker67)
  (clean shot139)
  (clean shot438)
  (empty shaker67)
  (empty shot139)
  (empty shot438)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker67 l0)
  (shaker-level shaker67 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail32 ingredient294)
  (cocktail-part2 cocktail32 ingredient489)
)
 (:goal
  (and
      (contains shot139 cocktail32)
)))
