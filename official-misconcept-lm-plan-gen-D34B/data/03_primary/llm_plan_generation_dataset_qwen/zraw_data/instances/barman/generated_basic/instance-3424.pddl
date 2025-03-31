(define (problem prob)
 (:domain barman)
 (:objects 
      shaker456 - shaker
      left right - hand
      shot335 shot345 - shot
      ingredient64 ingredient156 ingredient389 - ingredient
      cocktail206 - cocktail
      dispenser159 dispenser441 dispenser69 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker456)
  (ontable shot335)
  (ontable shot345)
  (dispenses dispenser159 ingredient64)
  (dispenses dispenser441 ingredient156)
  (dispenses dispenser69 ingredient389)
  (clean shaker456)
  (clean shot335)
  (clean shot345)
  (empty shaker456)
  (empty shot335)
  (empty shot345)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker456 l0)
  (shaker-level shaker456 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail206 ingredient64)
  (cocktail-part2 cocktail206 ingredient389)
)
 (:goal
  (and
      (contains shot335 cocktail206)
)))
