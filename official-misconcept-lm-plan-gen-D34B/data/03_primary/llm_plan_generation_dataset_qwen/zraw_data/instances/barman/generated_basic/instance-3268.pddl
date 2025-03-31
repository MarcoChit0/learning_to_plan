(define (problem prob)
 (:domain barman)
 (:objects 
      shaker277 - shaker
      left right - hand
      shot324 shot353 shot468 - shot
      ingredient30 ingredient198 ingredient386 ingredient398 - ingredient
      cocktail457 - cocktail
      dispenser268 dispenser167 dispenser461 dispenser266 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker277)
  (ontable shot324)
  (ontable shot353)
  (ontable shot468)
  (dispenses dispenser268 ingredient30)
  (dispenses dispenser167 ingredient198)
  (dispenses dispenser461 ingredient386)
  (dispenses dispenser266 ingredient398)
  (clean shaker277)
  (clean shot324)
  (clean shot353)
  (clean shot468)
  (empty shaker277)
  (empty shot324)
  (empty shot353)
  (empty shot468)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker277 l0)
  (shaker-level shaker277 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail457 ingredient398)
  (cocktail-part2 cocktail457 ingredient386)
)
 (:goal
  (and
      (contains shot324 cocktail457)
      (contains shot353 cocktail457)
)))
