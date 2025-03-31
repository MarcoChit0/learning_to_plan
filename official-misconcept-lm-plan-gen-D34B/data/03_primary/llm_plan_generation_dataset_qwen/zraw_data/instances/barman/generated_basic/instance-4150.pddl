(define (problem prob)
 (:domain barman)
 (:objects 
      shaker349 - shaker
      left right - hand
      shot274 - shot
      ingredient129 ingredient141 - ingredient
      cocktail55 - cocktail
      dispenser264 dispenser38 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker349)
  (ontable shot274)
  (dispenses dispenser264 ingredient129)
  (dispenses dispenser38 ingredient141)
  (clean shaker349)
  (clean shot274)
  (empty shaker349)
  (empty shot274)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker349 l0)
  (shaker-level shaker349 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail55 ingredient129)
  (cocktail-part2 cocktail55 ingredient141)
)
 (:goal
  (and
      (contains shot274 cocktail55)
)))
