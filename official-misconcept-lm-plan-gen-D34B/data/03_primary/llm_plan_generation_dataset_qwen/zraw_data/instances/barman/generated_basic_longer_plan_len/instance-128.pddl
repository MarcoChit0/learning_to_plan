(define (problem prob)
 (:domain barman)
 (:objects 
      shaker24 - shaker
      left right - hand
      shot441 shot345 shot189 - shot
      ingredient367 ingredient378 - ingredient
      cocktail1 cocktail2 - cocktail
      dispenser349 dispenser145 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker24)
  (ontable shot441)
  (ontable shot345)
  (ontable shot189)
  (dispenses dispenser349 ingredient367)
  (dispenses dispenser145 ingredient378)
  (clean shaker24)
  (clean shot441)
  (clean shot345)
  (clean shot189)
  (empty shaker24)
  (empty shot441)
  (empty shot345)
  (empty shot189)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker24 l0)
  (shaker-level shaker24 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient367)
  (cocktail-part2 cocktail1 ingredient378)
  (cocktail-part1 cocktail2 ingredient378)
  (cocktail-part2 cocktail2 ingredient367)
)
 (:goal
  (and
      (contains shot441 cocktail2)
      (contains shot345 cocktail1)
)))
