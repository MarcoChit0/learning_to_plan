(define (problem prob)
 (:domain barman)
 (:objects 
      shaker149 - shaker
      left right - hand
      shot181 shot91 shot99 - shot
      ingredient116 ingredient43 - ingredient
      cocktail409 - cocktail
      dispenser50 dispenser61 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker149)
  (ontable shot181)
  (ontable shot91)
  (ontable shot99)
  (dispenses dispenser50 ingredient116)
  (dispenses dispenser61 ingredient43)
  (clean shaker149)
  (clean shot181)
  (clean shot91)
  (clean shot99)
  (empty shaker149)
  (empty shot181)
  (empty shot91)
  (empty shot99)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker149 l0)
  (shaker-level shaker149 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail409 ingredient116)
  (cocktail-part2 cocktail409 ingredient43)
)
 (:goal
  (and
      (contains shot181 cocktail409)
      (contains shot91 cocktail409)
)))
