(define (problem prob)
 (:domain barman)
 (:objects 
      shaker317 - shaker
      left right - hand
      shot355 shot254 - shot
      ingredient270 ingredient491 ingredient336 ingredient268 - ingredient
      cocktail28 - cocktail
      dispenser442 dispenser328 dispenser121 dispenser41 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker317)
  (ontable shot355)
  (ontable shot254)
  (dispenses dispenser442 ingredient270)
  (dispenses dispenser328 ingredient491)
  (dispenses dispenser121 ingredient336)
  (dispenses dispenser41 ingredient268)
  (clean shaker317)
  (clean shot355)
  (clean shot254)
  (empty shaker317)
  (empty shot355)
  (empty shot254)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker317 l0)
  (shaker-level shaker317 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail28 ingredient491)
  (cocktail-part2 cocktail28 ingredient270)
)
 (:goal
  (and
      (contains shot355 cocktail28)
)))
