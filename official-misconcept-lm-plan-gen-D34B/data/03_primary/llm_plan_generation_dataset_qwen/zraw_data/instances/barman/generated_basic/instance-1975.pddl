(define (problem prob)
 (:domain barman)
 (:objects 
      shaker379 - shaker
      left right - hand
      shot36 shot417 - shot
      ingredient323 ingredient195 ingredient191 - ingredient
      cocktail236 - cocktail
      dispenser325 dispenser181 dispenser305 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker379)
  (ontable shot36)
  (ontable shot417)
  (dispenses dispenser325 ingredient323)
  (dispenses dispenser181 ingredient195)
  (dispenses dispenser305 ingredient191)
  (clean shaker379)
  (clean shot36)
  (clean shot417)
  (empty shaker379)
  (empty shot36)
  (empty shot417)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker379 l0)
  (shaker-level shaker379 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail236 ingredient323)
  (cocktail-part2 cocktail236 ingredient195)
)
 (:goal
  (and
      (contains shot36 cocktail236)
)))
