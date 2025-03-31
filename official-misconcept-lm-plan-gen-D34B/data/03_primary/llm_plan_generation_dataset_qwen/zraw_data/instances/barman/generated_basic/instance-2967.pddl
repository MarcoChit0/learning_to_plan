(define (problem prob)
 (:domain barman)
 (:objects 
      shaker365 - shaker
      left right - hand
      shot333 - shot
      ingredient288 ingredient75 - ingredient
      cocktail207 - cocktail
      dispenser273 dispenser470 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker365)
  (ontable shot333)
  (dispenses dispenser273 ingredient288)
  (dispenses dispenser470 ingredient75)
  (clean shaker365)
  (clean shot333)
  (empty shaker365)
  (empty shot333)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker365 l0)
  (shaker-level shaker365 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail207 ingredient75)
  (cocktail-part2 cocktail207 ingredient288)
)
 (:goal
  (and
      (contains shot333 cocktail207)
)))
