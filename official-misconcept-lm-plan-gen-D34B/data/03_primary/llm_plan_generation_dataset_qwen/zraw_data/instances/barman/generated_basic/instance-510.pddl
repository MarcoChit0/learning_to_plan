(define (problem prob)
 (:domain barman)
 (:objects 
      shaker459 - shaker
      left right - hand
      shot27 shot464 - shot
      ingredient125 ingredient485 ingredient272 ingredient95 - ingredient
      cocktail1 - cocktail
      dispenser436 dispenser154 dispenser174 dispenser273 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker459)
  (ontable shot27)
  (ontable shot464)
  (dispenses dispenser436 ingredient125)
  (dispenses dispenser154 ingredient485)
  (dispenses dispenser174 ingredient272)
  (dispenses dispenser273 ingredient95)
  (clean shaker459)
  (clean shot27)
  (clean shot464)
  (empty shaker459)
  (empty shot27)
  (empty shot464)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker459 l0)
  (shaker-level shaker459 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient95)
  (cocktail-part2 cocktail1 ingredient485)
)
 (:goal
  (and
      (contains shot27 cocktail1)
)))
