(define (problem prob)
 (:domain barman)
 (:objects 
      shaker486 - shaker
      left right - hand
      shot179 shot101 shot162 - shot
      ingredient225 ingredient59 - ingredient
      cocktail421 - cocktail
      dispenser486 dispenser446 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker486)
  (ontable shot179)
  (ontable shot101)
  (ontable shot162)
  (dispenses dispenser486 ingredient225)
  (dispenses dispenser446 ingredient59)
  (clean shaker486)
  (clean shot179)
  (clean shot101)
  (clean shot162)
  (empty shaker486)
  (empty shot179)
  (empty shot101)
  (empty shot162)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker486 l0)
  (shaker-level shaker486 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail421 ingredient225)
  (cocktail-part2 cocktail421 ingredient59)
)
 (:goal
  (and
      (contains shot179 cocktail421)
      (contains shot101 cocktail421)
)))
