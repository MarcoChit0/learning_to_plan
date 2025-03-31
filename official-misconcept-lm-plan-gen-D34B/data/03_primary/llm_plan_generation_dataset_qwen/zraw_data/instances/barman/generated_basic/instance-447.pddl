(define (problem prob)
 (:domain barman)
 (:objects 
      shaker25 - shaker
      left right - hand
      shot175 shot498 - shot
      ingredient151 ingredient97 ingredient316 - ingredient
      cocktail1 - cocktail
      dispenser487 dispenser440 dispenser340 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker25)
  (ontable shot175)
  (ontable shot498)
  (dispenses dispenser487 ingredient151)
  (dispenses dispenser440 ingredient97)
  (dispenses dispenser340 ingredient316)
  (clean shaker25)
  (clean shot175)
  (clean shot498)
  (empty shaker25)
  (empty shot175)
  (empty shot498)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker25 l0)
  (shaker-level shaker25 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient97)
  (cocktail-part2 cocktail1 ingredient316)
)
 (:goal
  (and
      (contains shot175 cocktail1)
)))
