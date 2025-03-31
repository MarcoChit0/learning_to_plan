(define (problem prob)
 (:domain barman)
 (:objects 
      shaker200 - shaker
      left right - hand
      shot138 shot111 shot267 - shot
      ingredient212 ingredient147 ingredient233 ingredient465 - ingredient
      cocktail116 - cocktail
      dispenser423 dispenser325 dispenser472 dispenser271 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker200)
  (ontable shot138)
  (ontable shot111)
  (ontable shot267)
  (dispenses dispenser423 ingredient212)
  (dispenses dispenser325 ingredient147)
  (dispenses dispenser472 ingredient233)
  (dispenses dispenser271 ingredient465)
  (clean shaker200)
  (clean shot138)
  (clean shot111)
  (clean shot267)
  (empty shaker200)
  (empty shot138)
  (empty shot111)
  (empty shot267)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker200 l0)
  (shaker-level shaker200 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail116 ingredient465)
  (cocktail-part2 cocktail116 ingredient233)
)
 (:goal
  (and
      (contains shot138 cocktail116)
      (contains shot111 ingredient233)
)))
