(define (problem prob)
 (:domain barman)
 (:objects 
      shaker251 - shaker
      left right - hand
      shot453 shot75 - shot
      ingredient361 ingredient455 - ingredient
      cocktail489 - cocktail
      dispenser453 dispenser479 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker251)
  (ontable shot453)
  (ontable shot75)
  (dispenses dispenser453 ingredient361)
  (dispenses dispenser479 ingredient455)
  (clean shaker251)
  (clean shot453)
  (clean shot75)
  (empty shaker251)
  (empty shot453)
  (empty shot75)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker251 l0)
  (shaker-level shaker251 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail489 ingredient361)
  (cocktail-part2 cocktail489 ingredient455)
)
 (:goal
  (and
      (contains shot453 cocktail489)
)))
