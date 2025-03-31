(define (problem prob)
 (:domain barman)
 (:objects 
      shaker376 - shaker
      left right - hand
      shot161 shot251 shot442 - shot
      ingredient486 ingredient210 ingredient70 ingredient243 - ingredient
      cocktail399 - cocktail
      dispenser226 dispenser198 dispenser454 dispenser71 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker376)
  (ontable shot161)
  (ontable shot251)
  (ontable shot442)
  (dispenses dispenser226 ingredient486)
  (dispenses dispenser198 ingredient210)
  (dispenses dispenser454 ingredient70)
  (dispenses dispenser71 ingredient243)
  (clean shaker376)
  (clean shot161)
  (clean shot251)
  (clean shot442)
  (empty shaker376)
  (empty shot161)
  (empty shot251)
  (empty shot442)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker376 l0)
  (shaker-level shaker376 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail399 ingredient486)
  (cocktail-part2 cocktail399 ingredient70)
)
 (:goal
  (and
      (contains shot161 cocktail399)
      (contains shot251 cocktail399)
)))
