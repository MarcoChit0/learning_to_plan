(define (problem prob)
 (:domain barman)
 (:objects 
      shaker107 - shaker
      left right - hand
      shot367 - shot
      ingredient139 ingredient217 ingredient224 - ingredient
      cocktail43 - cocktail
      dispenser450 dispenser151 dispenser441 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker107)
  (ontable shot367)
  (dispenses dispenser450 ingredient139)
  (dispenses dispenser151 ingredient217)
  (dispenses dispenser441 ingredient224)
  (clean shaker107)
  (clean shot367)
  (empty shaker107)
  (empty shot367)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker107 l0)
  (shaker-level shaker107 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail43 ingredient217)
  (cocktail-part2 cocktail43 ingredient224)
)
 (:goal
  (and
      (contains shot367 cocktail43)
)))
