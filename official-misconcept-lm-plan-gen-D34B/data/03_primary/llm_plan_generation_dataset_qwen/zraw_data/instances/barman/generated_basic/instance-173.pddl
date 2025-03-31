(define (problem prob)
 (:domain barman)
 (:objects 
      shaker314 - shaker
      left right - hand
      shot115 shot386 - shot
      ingredient168 ingredient254 ingredient296 - ingredient
      cocktail1 - cocktail
      dispenser203 dispenser134 dispenser43 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker314)
  (ontable shot115)
  (ontable shot386)
  (dispenses dispenser203 ingredient168)
  (dispenses dispenser134 ingredient254)
  (dispenses dispenser43 ingredient296)
  (clean shaker314)
  (clean shot115)
  (clean shot386)
  (empty shaker314)
  (empty shot115)
  (empty shot386)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker314 l0)
  (shaker-level shaker314 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient296)
  (cocktail-part2 cocktail1 ingredient254)
)
 (:goal
  (and
      (contains shot115 cocktail1)
)))
