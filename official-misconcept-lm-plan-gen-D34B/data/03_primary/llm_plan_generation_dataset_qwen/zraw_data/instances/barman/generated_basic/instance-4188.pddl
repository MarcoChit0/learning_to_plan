(define (problem prob)
 (:domain barman)
 (:objects 
      shaker117 - shaker
      left right - hand
      shot361 - shot
      ingredient282 ingredient346 ingredient303 ingredient203 - ingredient
      cocktail456 - cocktail
      dispenser134 dispenser94 dispenser165 dispenser425 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker117)
  (ontable shot361)
  (dispenses dispenser134 ingredient282)
  (dispenses dispenser94 ingredient346)
  (dispenses dispenser165 ingredient303)
  (dispenses dispenser425 ingredient203)
  (clean shaker117)
  (clean shot361)
  (empty shaker117)
  (empty shot361)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker117 l0)
  (shaker-level shaker117 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail456 ingredient203)
  (cocktail-part2 cocktail456 ingredient282)
)
 (:goal
  (and
      (contains shot361 cocktail456)
)))
