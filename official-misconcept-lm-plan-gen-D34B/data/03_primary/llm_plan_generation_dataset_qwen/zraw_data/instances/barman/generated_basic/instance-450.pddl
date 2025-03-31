(define (problem prob)
 (:domain barman)
 (:objects 
      shaker464 - shaker
      left right - hand
      shot55 shot290 - shot
      ingredient410 ingredient125 ingredient48 - ingredient
      cocktail1 - cocktail
      dispenser49 dispenser311 dispenser479 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker464)
  (ontable shot55)
  (ontable shot290)
  (dispenses dispenser49 ingredient410)
  (dispenses dispenser311 ingredient125)
  (dispenses dispenser479 ingredient48)
  (clean shaker464)
  (clean shot55)
  (clean shot290)
  (empty shaker464)
  (empty shot55)
  (empty shot290)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker464 l0)
  (shaker-level shaker464 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient125)
  (cocktail-part2 cocktail1 ingredient48)
)
 (:goal
  (and
      (contains shot55 cocktail1)
)))
