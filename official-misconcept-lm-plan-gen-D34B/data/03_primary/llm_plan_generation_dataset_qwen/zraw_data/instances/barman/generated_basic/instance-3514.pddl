(define (problem prob)
 (:domain barman)
 (:objects 
      shaker78 - shaker
      left right - hand
      shot201 shot386 - shot
      ingredient403 ingredient175 ingredient45 ingredient211 - ingredient
      cocktail464 - cocktail
      dispenser17 dispenser439 dispenser242 dispenser126 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker78)
  (ontable shot201)
  (ontable shot386)
  (dispenses dispenser17 ingredient403)
  (dispenses dispenser439 ingredient175)
  (dispenses dispenser242 ingredient45)
  (dispenses dispenser126 ingredient211)
  (clean shaker78)
  (clean shot201)
  (clean shot386)
  (empty shaker78)
  (empty shot201)
  (empty shot386)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker78 l0)
  (shaker-level shaker78 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail464 ingredient45)
  (cocktail-part2 cocktail464 ingredient175)
)
 (:goal
  (and
      (contains shot201 cocktail464)
)))
