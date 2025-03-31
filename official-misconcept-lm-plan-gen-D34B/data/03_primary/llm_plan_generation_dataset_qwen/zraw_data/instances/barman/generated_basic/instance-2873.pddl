(define (problem prob)
 (:domain barman)
 (:objects 
      shaker213 - shaker
      left right - hand
      shot408 shot247 - shot
      ingredient143 ingredient313 ingredient213 ingredient88 - ingredient
      cocktail81 - cocktail
      dispenser39 dispenser324 dispenser486 dispenser6 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker213)
  (ontable shot408)
  (ontable shot247)
  (dispenses dispenser39 ingredient143)
  (dispenses dispenser324 ingredient313)
  (dispenses dispenser486 ingredient213)
  (dispenses dispenser6 ingredient88)
  (clean shaker213)
  (clean shot408)
  (clean shot247)
  (empty shaker213)
  (empty shot408)
  (empty shot247)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker213 l0)
  (shaker-level shaker213 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail81 ingredient313)
  (cocktail-part2 cocktail81 ingredient143)
)
 (:goal
  (and
      (contains shot408 cocktail81)
)))
