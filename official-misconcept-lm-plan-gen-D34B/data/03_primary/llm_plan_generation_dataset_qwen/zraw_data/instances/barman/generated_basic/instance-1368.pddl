(define (problem prob)
 (:domain barman)
 (:objects 
      shaker151 - shaker
      left right - hand
      shot213 - shot
      ingredient177 ingredient292 ingredient119 - ingredient
      cocktail228 - cocktail
      dispenser437 dispenser325 dispenser423 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker151)
  (ontable shot213)
  (dispenses dispenser437 ingredient177)
  (dispenses dispenser325 ingredient292)
  (dispenses dispenser423 ingredient119)
  (clean shaker151)
  (clean shot213)
  (empty shaker151)
  (empty shot213)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker151 l0)
  (shaker-level shaker151 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail228 ingredient177)
  (cocktail-part2 cocktail228 ingredient119)
)
 (:goal
  (and
      (contains shot213 cocktail228)
)))
