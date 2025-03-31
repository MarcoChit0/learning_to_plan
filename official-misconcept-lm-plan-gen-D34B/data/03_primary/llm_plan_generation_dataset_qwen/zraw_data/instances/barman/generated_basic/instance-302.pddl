(define (problem prob)
 (:domain barman)
 (:objects 
      shaker400 - shaker
      left right - hand
      shot101 shot16 - shot
      ingredient37 ingredient391 - ingredient
      cocktail1 - cocktail
      dispenser105 dispenser29 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker400)
  (ontable shot101)
  (ontable shot16)
  (dispenses dispenser105 ingredient37)
  (dispenses dispenser29 ingredient391)
  (clean shaker400)
  (clean shot101)
  (clean shot16)
  (empty shaker400)
  (empty shot101)
  (empty shot16)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker400 l0)
  (shaker-level shaker400 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient391)
  (cocktail-part2 cocktail1 ingredient37)
)
 (:goal
  (and
      (contains shot101 cocktail1)
)))
