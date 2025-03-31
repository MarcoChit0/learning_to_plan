(define (problem prob)
 (:domain barman)
 (:objects 
      shaker164 - shaker
      left right - hand
      shot267 shot254 shot206 - shot
      ingredient125 ingredient463 - ingredient
      cocktail19 - cocktail
      dispenser56 dispenser5 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker164)
  (ontable shot267)
  (ontable shot254)
  (ontable shot206)
  (dispenses dispenser56 ingredient125)
  (dispenses dispenser5 ingredient463)
  (clean shaker164)
  (clean shot267)
  (clean shot254)
  (clean shot206)
  (empty shaker164)
  (empty shot267)
  (empty shot254)
  (empty shot206)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker164 l0)
  (shaker-level shaker164 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail19 ingredient463)
  (cocktail-part2 cocktail19 ingredient125)
)
 (:goal
  (and
      (contains shot267 cocktail19)
      (contains shot254 cocktail19)
)))
