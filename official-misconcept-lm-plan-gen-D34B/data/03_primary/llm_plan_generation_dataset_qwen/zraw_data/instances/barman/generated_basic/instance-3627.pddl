(define (problem prob)
 (:domain barman)
 (:objects 
      shaker448 - shaker
      left right - hand
      shot467 - shot
      ingredient158 ingredient178 ingredient77 ingredient57 - ingredient
      cocktail449 - cocktail
      dispenser31 dispenser340 dispenser17 dispenser243 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker448)
  (ontable shot467)
  (dispenses dispenser31 ingredient158)
  (dispenses dispenser340 ingredient178)
  (dispenses dispenser17 ingredient77)
  (dispenses dispenser243 ingredient57)
  (clean shaker448)
  (clean shot467)
  (empty shaker448)
  (empty shot467)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker448 l0)
  (shaker-level shaker448 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail449 ingredient178)
  (cocktail-part2 cocktail449 ingredient77)
)
 (:goal
  (and
      (contains shot467 cocktail449)
)))
