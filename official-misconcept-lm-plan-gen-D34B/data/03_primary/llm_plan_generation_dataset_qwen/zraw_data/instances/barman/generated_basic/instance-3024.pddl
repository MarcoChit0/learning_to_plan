(define (problem prob)
 (:domain barman)
 (:objects 
      shaker392 - shaker
      left right - hand
      shot250 - shot
      ingredient84 ingredient498 ingredient317 - ingredient
      cocktail108 - cocktail
      dispenser290 dispenser57 dispenser89 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker392)
  (ontable shot250)
  (dispenses dispenser290 ingredient84)
  (dispenses dispenser57 ingredient498)
  (dispenses dispenser89 ingredient317)
  (clean shaker392)
  (clean shot250)
  (empty shaker392)
  (empty shot250)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker392 l0)
  (shaker-level shaker392 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail108 ingredient498)
  (cocktail-part2 cocktail108 ingredient317)
)
 (:goal
  (and
      (contains shot250 cocktail108)
)))
