(define (problem prob)
 (:domain barman)
 (:objects 
      shaker450 - shaker
      left right - hand
      shot162 shot361 - shot
      ingredient119 ingredient76 ingredient447 ingredient389 - ingredient
      cocktail421 - cocktail
      dispenser331 dispenser34 dispenser17 dispenser333 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker450)
  (ontable shot162)
  (ontable shot361)
  (dispenses dispenser331 ingredient119)
  (dispenses dispenser34 ingredient76)
  (dispenses dispenser17 ingredient447)
  (dispenses dispenser333 ingredient389)
  (clean shaker450)
  (clean shot162)
  (clean shot361)
  (empty shaker450)
  (empty shot162)
  (empty shot361)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker450 l0)
  (shaker-level shaker450 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail421 ingredient119)
  (cocktail-part2 cocktail421 ingredient447)
)
 (:goal
  (and
      (contains shot162 cocktail421)
)))
