(define (problem prob)
 (:domain barman)
 (:objects 
      shaker401 - shaker
      left right - hand
      shot416 shot30 shot98 shot432 - shot
      ingredient205 ingredient231 ingredient173 - ingredient
      cocktail322 - cocktail
      dispenser472 dispenser118 dispenser41 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker401)
  (ontable shot416)
  (ontable shot30)
  (ontable shot98)
  (ontable shot432)
  (dispenses dispenser472 ingredient205)
  (dispenses dispenser118 ingredient231)
  (dispenses dispenser41 ingredient173)
  (clean shaker401)
  (clean shot416)
  (clean shot30)
  (clean shot98)
  (clean shot432)
  (empty shaker401)
  (empty shot416)
  (empty shot30)
  (empty shot98)
  (empty shot432)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker401 l0)
  (shaker-level shaker401 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail322 ingredient231)
  (cocktail-part2 cocktail322 ingredient173)
)
 (:goal
  (and
      (contains shot416 cocktail322)
      (contains shot30 ingredient231)
      (contains shot98 cocktail322)
)))
