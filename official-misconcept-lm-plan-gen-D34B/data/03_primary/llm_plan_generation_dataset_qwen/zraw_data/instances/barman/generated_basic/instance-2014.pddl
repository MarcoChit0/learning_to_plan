(define (problem prob)
 (:domain barman)
 (:objects 
      shaker375 - shaker
      left right - hand
      shot92 shot105 shot115 - shot
      ingredient38 ingredient353 ingredient69 - ingredient
      cocktail131 - cocktail
      dispenser441 dispenser458 dispenser163 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker375)
  (ontable shot92)
  (ontable shot105)
  (ontable shot115)
  (dispenses dispenser441 ingredient38)
  (dispenses dispenser458 ingredient353)
  (dispenses dispenser163 ingredient69)
  (clean shaker375)
  (clean shot92)
  (clean shot105)
  (clean shot115)
  (empty shaker375)
  (empty shot92)
  (empty shot105)
  (empty shot115)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker375 l0)
  (shaker-level shaker375 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail131 ingredient38)
  (cocktail-part2 cocktail131 ingredient353)
)
 (:goal
  (and
      (contains shot92 cocktail131)
      (contains shot105 cocktail131)
)))
