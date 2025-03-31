(define (problem prob)
 (:domain barman)
 (:objects 
      shaker126 - shaker
      left right - hand
      shot438 - shot
      ingredient234 ingredient461 ingredient398 ingredient210 - ingredient
      cocktail261 - cocktail
      dispenser81 dispenser466 dispenser427 dispenser433 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker126)
  (ontable shot438)
  (dispenses dispenser81 ingredient234)
  (dispenses dispenser466 ingredient461)
  (dispenses dispenser427 ingredient398)
  (dispenses dispenser433 ingredient210)
  (clean shaker126)
  (clean shot438)
  (empty shaker126)
  (empty shot438)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker126 l0)
  (shaker-level shaker126 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail261 ingredient398)
  (cocktail-part2 cocktail261 ingredient210)
)
 (:goal
  (and
      (contains shot438 cocktail261)
)))
