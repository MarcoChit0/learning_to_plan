(define (problem prob)
 (:domain barman)
 (:objects 
      shaker436 - shaker
      left right - hand
      shot4 shot66 - shot
      ingredient313 ingredient102 ingredient316 ingredient304 - ingredient
      cocktail272 - cocktail
      dispenser445 dispenser101 dispenser23 dispenser365 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker436)
  (ontable shot4)
  (ontable shot66)
  (dispenses dispenser445 ingredient313)
  (dispenses dispenser101 ingredient102)
  (dispenses dispenser23 ingredient316)
  (dispenses dispenser365 ingredient304)
  (clean shaker436)
  (clean shot4)
  (clean shot66)
  (empty shaker436)
  (empty shot4)
  (empty shot66)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker436 l0)
  (shaker-level shaker436 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail272 ingredient304)
  (cocktail-part2 cocktail272 ingredient313)
)
 (:goal
  (and
      (contains shot4 cocktail272)
)))
