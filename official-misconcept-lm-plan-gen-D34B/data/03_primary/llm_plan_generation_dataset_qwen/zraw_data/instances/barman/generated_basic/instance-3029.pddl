(define (problem prob)
 (:domain barman)
 (:objects 
      shaker225 - shaker
      left right - hand
      shot289 - shot
      ingredient277 ingredient47 ingredient316 - ingredient
      cocktail233 - cocktail
      dispenser139 dispenser97 dispenser204 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker225)
  (ontable shot289)
  (dispenses dispenser139 ingredient277)
  (dispenses dispenser97 ingredient47)
  (dispenses dispenser204 ingredient316)
  (clean shaker225)
  (clean shot289)
  (empty shaker225)
  (empty shot289)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker225 l0)
  (shaker-level shaker225 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail233 ingredient277)
  (cocktail-part2 cocktail233 ingredient47)
)
 (:goal
  (and
      (contains shot289 cocktail233)
)))
