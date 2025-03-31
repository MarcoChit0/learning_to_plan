(define (problem prob)
 (:domain barman)
 (:objects 
      shaker391 - shaker
      left right - hand
      shot162 shot47 shot467 - shot
      ingredient387 ingredient253 ingredient436 ingredient360 - ingredient
      cocktail157 - cocktail
      dispenser331 dispenser159 dispenser483 dispenser431 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker391)
  (ontable shot162)
  (ontable shot47)
  (ontable shot467)
  (dispenses dispenser331 ingredient387)
  (dispenses dispenser159 ingredient253)
  (dispenses dispenser483 ingredient436)
  (dispenses dispenser431 ingredient360)
  (clean shaker391)
  (clean shot162)
  (clean shot47)
  (clean shot467)
  (empty shaker391)
  (empty shot162)
  (empty shot47)
  (empty shot467)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker391 l0)
  (shaker-level shaker391 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail157 ingredient360)
  (cocktail-part2 cocktail157 ingredient387)
)
 (:goal
  (and
      (contains shot162 cocktail157)
      (contains shot47 ingredient360)
)))
