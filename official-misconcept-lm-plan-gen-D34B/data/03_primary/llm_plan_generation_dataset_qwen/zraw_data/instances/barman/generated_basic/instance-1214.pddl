(define (problem prob)
 (:domain barman)
 (:objects 
      shaker431 - shaker
      left right - hand
      shot431 shot75 shot77 - shot
      ingredient60 ingredient245 ingredient308 - ingredient
      cocktail204 - cocktail
      dispenser212 dispenser313 dispenser76 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker431)
  (ontable shot431)
  (ontable shot75)
  (ontable shot77)
  (dispenses dispenser212 ingredient60)
  (dispenses dispenser313 ingredient245)
  (dispenses dispenser76 ingredient308)
  (clean shaker431)
  (clean shot431)
  (clean shot75)
  (clean shot77)
  (empty shaker431)
  (empty shot431)
  (empty shot75)
  (empty shot77)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker431 l0)
  (shaker-level shaker431 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail204 ingredient308)
  (cocktail-part2 cocktail204 ingredient245)
)
 (:goal
  (and
      (contains shot431 cocktail204)
      (contains shot75 ingredient308)
)))
