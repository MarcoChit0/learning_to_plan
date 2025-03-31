(define (problem prob)
 (:domain barman)
 (:objects 
      shaker258 - shaker
      left right - hand
      shot28 - shot
      ingredient89 ingredient76 - ingredient
      cocktail62 - cocktail
      dispenser440 dispenser363 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker258)
  (ontable shot28)
  (dispenses dispenser440 ingredient89)
  (dispenses dispenser363 ingredient76)
  (clean shaker258)
  (clean shot28)
  (empty shaker258)
  (empty shot28)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker258 l0)
  (shaker-level shaker258 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail62 ingredient76)
  (cocktail-part2 cocktail62 ingredient89)
)
 (:goal
  (and
      (contains shot28 cocktail62)
)))
