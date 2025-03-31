(define (problem prob)
 (:domain barman)
 (:objects 
      shaker254 - shaker
      left right - hand
      shot136 shot492 shot192 - shot
      ingredient358 ingredient431 - ingredient
      cocktail1 cocktail2 - cocktail
      dispenser216 dispenser22 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker254)
  (ontable shot136)
  (ontable shot492)
  (ontable shot192)
  (dispenses dispenser216 ingredient358)
  (dispenses dispenser22 ingredient431)
  (clean shaker254)
  (clean shot136)
  (clean shot492)
  (clean shot192)
  (empty shaker254)
  (empty shot136)
  (empty shot492)
  (empty shot192)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker254 l0)
  (shaker-level shaker254 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient431)
  (cocktail-part2 cocktail1 ingredient358)
  (cocktail-part1 cocktail2 ingredient431)
  (cocktail-part2 cocktail2 ingredient358)
)
 (:goal
  (and
      (contains shot136 cocktail1)
      (contains shot492 cocktail2)
)))
