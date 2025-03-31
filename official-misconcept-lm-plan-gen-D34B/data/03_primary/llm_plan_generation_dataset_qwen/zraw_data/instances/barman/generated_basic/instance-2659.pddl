(define (problem prob)
 (:domain barman)
 (:objects 
      shaker5 - shaker
      left right - hand
      shot166 - shot
      ingredient184 ingredient451 ingredient346 ingredient251 - ingredient
      cocktail242 - cocktail
      dispenser202 dispenser210 dispenser375 dispenser471 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker5)
  (ontable shot166)
  (dispenses dispenser202 ingredient184)
  (dispenses dispenser210 ingredient451)
  (dispenses dispenser375 ingredient346)
  (dispenses dispenser471 ingredient251)
  (clean shaker5)
  (clean shot166)
  (empty shaker5)
  (empty shot166)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker5 l0)
  (shaker-level shaker5 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail242 ingredient451)
  (cocktail-part2 cocktail242 ingredient346)
)
 (:goal
  (and
      (contains shot166 cocktail242)
)))
