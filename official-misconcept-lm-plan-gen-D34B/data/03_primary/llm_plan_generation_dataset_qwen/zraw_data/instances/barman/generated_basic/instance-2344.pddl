(define (problem prob)
 (:domain barman)
 (:objects 
      shaker365 - shaker
      left right - hand
      shot480 shot345 - shot
      ingredient213 ingredient157 ingredient175 ingredient369 - ingredient
      cocktail110 - cocktail
      dispenser379 dispenser454 dispenser309 dispenser496 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker365)
  (ontable shot480)
  (ontable shot345)
  (dispenses dispenser379 ingredient213)
  (dispenses dispenser454 ingredient157)
  (dispenses dispenser309 ingredient175)
  (dispenses dispenser496 ingredient369)
  (clean shaker365)
  (clean shot480)
  (clean shot345)
  (empty shaker365)
  (empty shot480)
  (empty shot345)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker365 l0)
  (shaker-level shaker365 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail110 ingredient175)
  (cocktail-part2 cocktail110 ingredient157)
)
 (:goal
  (and
      (contains shot480 cocktail110)
)))
