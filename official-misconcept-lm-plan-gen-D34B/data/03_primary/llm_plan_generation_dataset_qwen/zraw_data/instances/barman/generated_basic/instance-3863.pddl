(define (problem prob)
 (:domain barman)
 (:objects 
      shaker12 - shaker
      left right - hand
      shot145 shot101 - shot
      ingredient197 ingredient252 ingredient477 - ingredient
      cocktail216 - cocktail
      dispenser431 dispenser134 dispenser6 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker12)
  (ontable shot145)
  (ontable shot101)
  (dispenses dispenser431 ingredient197)
  (dispenses dispenser134 ingredient252)
  (dispenses dispenser6 ingredient477)
  (clean shaker12)
  (clean shot145)
  (clean shot101)
  (empty shaker12)
  (empty shot145)
  (empty shot101)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker12 l0)
  (shaker-level shaker12 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail216 ingredient252)
  (cocktail-part2 cocktail216 ingredient477)
)
 (:goal
  (and
      (contains shot145 cocktail216)
)))
