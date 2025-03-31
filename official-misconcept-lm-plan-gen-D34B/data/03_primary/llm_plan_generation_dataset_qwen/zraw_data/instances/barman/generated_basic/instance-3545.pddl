(define (problem prob)
 (:domain barman)
 (:objects 
      shaker397 - shaker
      left right - hand
      shot97 - shot
      ingredient134 ingredient224 ingredient354 - ingredient
      cocktail39 - cocktail
      dispenser6 dispenser448 dispenser407 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker397)
  (ontable shot97)
  (dispenses dispenser6 ingredient134)
  (dispenses dispenser448 ingredient224)
  (dispenses dispenser407 ingredient354)
  (clean shaker397)
  (clean shot97)
  (empty shaker397)
  (empty shot97)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker397 l0)
  (shaker-level shaker397 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail39 ingredient224)
  (cocktail-part2 cocktail39 ingredient134)
)
 (:goal
  (and
      (contains shot97 cocktail39)
)))
