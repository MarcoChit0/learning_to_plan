(define (problem prob)
 (:domain barman)
 (:objects 
      shaker293 - shaker
      left right - hand
      shot228 shot99 - shot
      ingredient342 ingredient321 - ingredient
      cocktail1 cocktail2 - cocktail
      dispenser39 dispenser200 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker293)
  (ontable shot228)
  (ontable shot99)
  (dispenses dispenser39 ingredient342)
  (dispenses dispenser200 ingredient321)
  (clean shaker293)
  (clean shot228)
  (clean shot99)
  (empty shaker293)
  (empty shot228)
  (empty shot99)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker293 l0)
  (shaker-level shaker293 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient342)
  (cocktail-part2 cocktail1 ingredient321)
  (cocktail-part1 cocktail2 ingredient321)
  (cocktail-part2 cocktail2 ingredient342)
)
 (:goal
  (and
      (contains shot228 cocktail1)
      (contains shot99 cocktail2)
)))
