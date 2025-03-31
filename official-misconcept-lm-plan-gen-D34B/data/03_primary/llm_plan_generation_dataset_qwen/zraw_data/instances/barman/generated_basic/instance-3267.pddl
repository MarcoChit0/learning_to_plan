(define (problem prob)
 (:domain barman)
 (:objects 
      shaker453 - shaker
      left right - hand
      shot18 shot288 shot225 - shot
      ingredient443 ingredient5 ingredient332 ingredient47 - ingredient
      cocktail252 - cocktail
      dispenser385 dispenser441 dispenser204 dispenser274 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker453)
  (ontable shot18)
  (ontable shot288)
  (ontable shot225)
  (dispenses dispenser385 ingredient443)
  (dispenses dispenser441 ingredient5)
  (dispenses dispenser204 ingredient332)
  (dispenses dispenser274 ingredient47)
  (clean shaker453)
  (clean shot18)
  (clean shot288)
  (clean shot225)
  (empty shaker453)
  (empty shot18)
  (empty shot288)
  (empty shot225)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker453 l0)
  (shaker-level shaker453 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail252 ingredient5)
  (cocktail-part2 cocktail252 ingredient332)
)
 (:goal
  (and
      (contains shot18 cocktail252)
      (contains shot288 cocktail252)
)))
