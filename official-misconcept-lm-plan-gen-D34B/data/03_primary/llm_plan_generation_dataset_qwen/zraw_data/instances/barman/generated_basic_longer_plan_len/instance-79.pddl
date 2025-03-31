(define (problem prob)
 (:domain barman)
 (:objects 
      shaker106 - shaker
      left right - hand
      shot369 shot382 shot254 - shot
      ingredient22 ingredient16 ingredient333 - ingredient
      cocktail1 cocktail2 - cocktail
      dispenser268 dispenser172 dispenser5 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker106)
  (ontable shot369)
  (ontable shot382)
  (ontable shot254)
  (dispenses dispenser268 ingredient22)
  (dispenses dispenser172 ingredient16)
  (dispenses dispenser5 ingredient333)
  (clean shaker106)
  (clean shot369)
  (clean shot382)
  (clean shot254)
  (empty shaker106)
  (empty shot369)
  (empty shot382)
  (empty shot254)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker106 l0)
  (shaker-level shaker106 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient16)
  (cocktail-part2 cocktail1 ingredient22)
  (cocktail-part1 cocktail2 ingredient333)
  (cocktail-part2 cocktail2 ingredient16)
)
 (:goal
  (and
      (contains shot369 cocktail1)
      (contains shot382 cocktail2)
)))
