(define (problem prob)
 (:domain barman)
 (:objects 
      shaker180 - shaker
      left right - hand
      shot365 - shot
      ingredient467 ingredient161 ingredient334 - ingredient
      cocktail443 - cocktail
      dispenser108 dispenser315 dispenser83 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker180)
  (ontable shot365)
  (dispenses dispenser108 ingredient467)
  (dispenses dispenser315 ingredient161)
  (dispenses dispenser83 ingredient334)
  (clean shaker180)
  (clean shot365)
  (empty shaker180)
  (empty shot365)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker180 l0)
  (shaker-level shaker180 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail443 ingredient161)
  (cocktail-part2 cocktail443 ingredient334)
)
 (:goal
  (and
      (contains shot365 cocktail443)
)))
