(define (problem prob)
 (:domain barman)
 (:objects 
      shaker332 - shaker
      left right - hand
      shot226 shot375 - shot
      ingredient109 ingredient396 ingredient105 - ingredient
      cocktail402 - cocktail
      dispenser144 dispenser249 dispenser295 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker332)
  (ontable shot226)
  (ontable shot375)
  (dispenses dispenser144 ingredient109)
  (dispenses dispenser249 ingredient396)
  (dispenses dispenser295 ingredient105)
  (clean shaker332)
  (clean shot226)
  (clean shot375)
  (empty shaker332)
  (empty shot226)
  (empty shot375)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker332 l0)
  (shaker-level shaker332 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail402 ingredient109)
  (cocktail-part2 cocktail402 ingredient105)
)
 (:goal
  (and
      (contains shot226 cocktail402)
)))
