(define (problem prob)
 (:domain barman)
 (:objects 
      shaker32 - shaker
      left right - hand
      shot497 shot51 shot182 - shot
      ingredient125 ingredient11 - ingredient
      cocktail1 cocktail2 - cocktail
      dispenser294 dispenser248 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker32)
  (ontable shot497)
  (ontable shot51)
  (ontable shot182)
  (dispenses dispenser294 ingredient125)
  (dispenses dispenser248 ingredient11)
  (clean shaker32)
  (clean shot497)
  (clean shot51)
  (clean shot182)
  (empty shaker32)
  (empty shot497)
  (empty shot51)
  (empty shot182)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker32 l0)
  (shaker-level shaker32 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient11)
  (cocktail-part2 cocktail1 ingredient125)
  (cocktail-part1 cocktail2 ingredient11)
  (cocktail-part2 cocktail2 ingredient125)
)
 (:goal
  (and
      (contains shot497 cocktail1)
      (contains shot51 cocktail2)
)))
