(define (problem prob)
 (:domain barman)
 (:objects 
      shaker163 - shaker
      left right - hand
      shot51 shot466 shot21 - shot
      ingredient407 ingredient352 - ingredient
      cocktail245 - cocktail
      dispenser117 dispenser144 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker163)
  (ontable shot51)
  (ontable shot466)
  (ontable shot21)
  (dispenses dispenser117 ingredient407)
  (dispenses dispenser144 ingredient352)
  (clean shaker163)
  (clean shot51)
  (clean shot466)
  (clean shot21)
  (empty shaker163)
  (empty shot51)
  (empty shot466)
  (empty shot21)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker163 l0)
  (shaker-level shaker163 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail245 ingredient407)
  (cocktail-part2 cocktail245 ingredient352)
)
 (:goal
  (and
      (contains shot51 cocktail245)
      (contains shot466 ingredient352)
)))
