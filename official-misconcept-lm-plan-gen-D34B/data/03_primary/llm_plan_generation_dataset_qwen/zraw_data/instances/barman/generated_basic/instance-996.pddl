(define (problem prob)
 (:domain barman)
 (:objects 
      shaker216 - shaker
      left right - hand
      shot324 shot274 - shot
      ingredient254 ingredient33 - ingredient
      cocktail1 - cocktail
      dispenser67 dispenser204 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker216)
  (ontable shot324)
  (ontable shot274)
  (dispenses dispenser67 ingredient254)
  (dispenses dispenser204 ingredient33)
  (clean shaker216)
  (clean shot324)
  (clean shot274)
  (empty shaker216)
  (empty shot324)
  (empty shot274)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker216 l0)
  (shaker-level shaker216 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient254)
  (cocktail-part2 cocktail1 ingredient33)
)
 (:goal
  (and
      (contains shot324 cocktail1)
)))
