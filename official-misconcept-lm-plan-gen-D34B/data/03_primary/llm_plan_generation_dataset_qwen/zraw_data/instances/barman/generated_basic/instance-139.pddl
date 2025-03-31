(define (problem prob)
 (:domain barman)
 (:objects 
      shaker259 - shaker
      left right - hand
      shot252 - shot
      ingredient131 ingredient231 ingredient291 ingredient484 - ingredient
      cocktail1 - cocktail
      dispenser470 dispenser325 dispenser434 dispenser95 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker259)
  (ontable shot252)
  (dispenses dispenser470 ingredient131)
  (dispenses dispenser325 ingredient231)
  (dispenses dispenser434 ingredient291)
  (dispenses dispenser95 ingredient484)
  (clean shaker259)
  (clean shot252)
  (empty shaker259)
  (empty shot252)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker259 l0)
  (shaker-level shaker259 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient484)
  (cocktail-part2 cocktail1 ingredient231)
)
 (:goal
  (and
      (contains shot252 cocktail1)
)))
