(define (problem prob)
 (:domain barman)
 (:objects 
      shaker190 - shaker
      left right - hand
      shot441 shot239 - shot
      ingredient198 ingredient271 ingredient408 - ingredient
      cocktail111 - cocktail
      dispenser204 dispenser479 dispenser145 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker190)
  (ontable shot441)
  (ontable shot239)
  (dispenses dispenser204 ingredient198)
  (dispenses dispenser479 ingredient271)
  (dispenses dispenser145 ingredient408)
  (clean shaker190)
  (clean shot441)
  (clean shot239)
  (empty shaker190)
  (empty shot441)
  (empty shot239)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker190 l0)
  (shaker-level shaker190 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail111 ingredient408)
  (cocktail-part2 cocktail111 ingredient271)
)
 (:goal
  (and
      (contains shot441 cocktail111)
)))
