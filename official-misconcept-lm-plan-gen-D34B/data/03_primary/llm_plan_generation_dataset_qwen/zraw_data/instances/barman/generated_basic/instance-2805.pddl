(define (problem prob)
 (:domain barman)
 (:objects 
      shaker317 - shaker
      left right - hand
      shot480 shot376 - shot
      ingredient263 ingredient353 - ingredient
      cocktail353 - cocktail
      dispenser428 dispenser473 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker317)
  (ontable shot480)
  (ontable shot376)
  (dispenses dispenser428 ingredient263)
  (dispenses dispenser473 ingredient353)
  (clean shaker317)
  (clean shot480)
  (clean shot376)
  (empty shaker317)
  (empty shot480)
  (empty shot376)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker317 l0)
  (shaker-level shaker317 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail353 ingredient353)
  (cocktail-part2 cocktail353 ingredient263)
)
 (:goal
  (and
      (contains shot480 cocktail353)
)))
