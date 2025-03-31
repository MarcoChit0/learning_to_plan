(define (problem prob)
 (:domain barman)
 (:objects 
      shaker111 - shaker
      left right - hand
      shot314 shot190 - shot
      ingredient321 ingredient467 ingredient195 - ingredient
      cocktail289 - cocktail
      dispenser100 dispenser153 dispenser292 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker111)
  (ontable shot314)
  (ontable shot190)
  (dispenses dispenser100 ingredient321)
  (dispenses dispenser153 ingredient467)
  (dispenses dispenser292 ingredient195)
  (clean shaker111)
  (clean shot314)
  (clean shot190)
  (empty shaker111)
  (empty shot314)
  (empty shot190)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker111 l0)
  (shaker-level shaker111 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail289 ingredient467)
  (cocktail-part2 cocktail289 ingredient321)
)
 (:goal
  (and
      (contains shot314 cocktail289)
)))
