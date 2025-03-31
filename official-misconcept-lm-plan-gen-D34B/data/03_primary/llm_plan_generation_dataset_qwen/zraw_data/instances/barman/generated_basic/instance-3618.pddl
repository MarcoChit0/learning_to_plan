(define (problem prob)
 (:domain barman)
 (:objects 
      shaker304 - shaker
      left right - hand
      shot280 shot446 shot197 - shot
      ingredient326 ingredient181 ingredient411 - ingredient
      cocktail271 - cocktail
      dispenser96 dispenser42 dispenser79 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker304)
  (ontable shot280)
  (ontable shot446)
  (ontable shot197)
  (dispenses dispenser96 ingredient326)
  (dispenses dispenser42 ingredient181)
  (dispenses dispenser79 ingredient411)
  (clean shaker304)
  (clean shot280)
  (clean shot446)
  (clean shot197)
  (empty shaker304)
  (empty shot280)
  (empty shot446)
  (empty shot197)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker304 l0)
  (shaker-level shaker304 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail271 ingredient181)
  (cocktail-part2 cocktail271 ingredient326)
)
 (:goal
  (and
      (contains shot280 cocktail271)
      (contains shot446 ingredient411)
)))
