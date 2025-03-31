(define (problem prob)
 (:domain barman)
 (:objects 
      shaker461 - shaker
      left right - hand
      shot263 shot259 - shot
      ingredient175 ingredient298 ingredient346 - ingredient
      cocktail114 - cocktail
      dispenser356 dispenser98 dispenser306 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker461)
  (ontable shot263)
  (ontable shot259)
  (dispenses dispenser356 ingredient175)
  (dispenses dispenser98 ingredient298)
  (dispenses dispenser306 ingredient346)
  (clean shaker461)
  (clean shot263)
  (clean shot259)
  (empty shaker461)
  (empty shot263)
  (empty shot259)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker461 l0)
  (shaker-level shaker461 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail114 ingredient298)
  (cocktail-part2 cocktail114 ingredient175)
)
 (:goal
  (and
      (contains shot263 cocktail114)
)))
