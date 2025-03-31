(define (problem prob)
 (:domain barman)
 (:objects 
      shaker75 - shaker
      left right - hand
      shot476 shot72 - shot
      ingredient157 ingredient52 ingredient420 - ingredient
      cocktail245 - cocktail
      dispenser451 dispenser309 dispenser467 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker75)
  (ontable shot476)
  (ontable shot72)
  (dispenses dispenser451 ingredient157)
  (dispenses dispenser309 ingredient52)
  (dispenses dispenser467 ingredient420)
  (clean shaker75)
  (clean shot476)
  (clean shot72)
  (empty shaker75)
  (empty shot476)
  (empty shot72)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker75 l0)
  (shaker-level shaker75 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail245 ingredient52)
  (cocktail-part2 cocktail245 ingredient157)
)
 (:goal
  (and
      (contains shot476 cocktail245)
)))
