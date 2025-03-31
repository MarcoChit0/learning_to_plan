(define (problem prob)
 (:domain barman)
 (:objects 
      shaker149 - shaker
      left right - hand
      shot480 shot313 - shot
      ingredient67 ingredient64 ingredient52 ingredient230 - ingredient
      cocktail195 - cocktail
      dispenser182 dispenser382 dispenser493 dispenser175 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker149)
  (ontable shot480)
  (ontable shot313)
  (dispenses dispenser182 ingredient67)
  (dispenses dispenser382 ingredient64)
  (dispenses dispenser493 ingredient52)
  (dispenses dispenser175 ingredient230)
  (clean shaker149)
  (clean shot480)
  (clean shot313)
  (empty shaker149)
  (empty shot480)
  (empty shot313)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker149 l0)
  (shaker-level shaker149 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail195 ingredient230)
  (cocktail-part2 cocktail195 ingredient67)
)
 (:goal
  (and
      (contains shot480 cocktail195)
)))
