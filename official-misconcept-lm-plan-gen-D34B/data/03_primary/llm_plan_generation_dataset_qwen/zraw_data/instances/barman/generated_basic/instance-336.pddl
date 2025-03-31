(define (problem prob)
 (:domain barman)
 (:objects 
      shaker177 - shaker
      left right - hand
      shot390 - shot
      ingredient243 ingredient464 - ingredient
      cocktail1 - cocktail
      dispenser483 dispenser93 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker177)
  (ontable shot390)
  (dispenses dispenser483 ingredient243)
  (dispenses dispenser93 ingredient464)
  (clean shaker177)
  (clean shot390)
  (empty shaker177)
  (empty shot390)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker177 l0)
  (shaker-level shaker177 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient464)
  (cocktail-part2 cocktail1 ingredient243)
)
 (:goal
  (and
      (contains shot390 cocktail1)
)))
