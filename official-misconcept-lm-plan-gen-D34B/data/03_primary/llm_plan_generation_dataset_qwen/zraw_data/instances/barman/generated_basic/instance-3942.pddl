(define (problem prob)
 (:domain barman)
 (:objects 
      shaker73 - shaker
      left right - hand
      shot238 shot328 - shot
      ingredient361 ingredient40 - ingredient
      cocktail327 - cocktail
      dispenser54 dispenser342 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker73)
  (ontable shot238)
  (ontable shot328)
  (dispenses dispenser54 ingredient361)
  (dispenses dispenser342 ingredient40)
  (clean shaker73)
  (clean shot238)
  (clean shot328)
  (empty shaker73)
  (empty shot238)
  (empty shot328)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker73 l0)
  (shaker-level shaker73 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail327 ingredient40)
  (cocktail-part2 cocktail327 ingredient361)
)
 (:goal
  (and
      (contains shot238 cocktail327)
)))
